#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(sigfit)
    library(rstan)
})

parse_args <- function(args) {
    out <- list(
        root = "/work",
        analysis_root = "/work/sigfit_poulos2018_20260726",
        workers = 64L,
        chains = 4L,
        iter = 2000L,
        warmup = 1000L,
        thin = 1L,
        base_seed = 2018L,
        max_tasks = 0L,
        dataset = "ALL",
        cohort = "ALL"
    )
    i <- 1L
    while (i <= length(args)) {
        key <- sub("^--", "", args[[i]])
        if (i == length(args)) stop("Missing value for --", key)
        value <- args[[i + 1L]]
        key <- gsub("-", "_", key, fixed = TRUE)
        if (!key %in% names(out)) stop("Unknown argument: --", key)
        out[[key]] <- value
        i <- i + 2L
    }
    integer_keys <- c("workers", "chains", "iter", "warmup", "thin", "base_seed", "max_tasks")
    for (key in integer_keys) out[[key]] <- as.integer(out[[key]])
    out
}

atomic_write_table <- function(x, path) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    tmp <- paste0(path, ".tmp.", Sys.getpid())
    write.table(x, tmp, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE, na = "NA")
    if (!file.rename(tmp, path)) {
        unlink(tmp)
        stop("Could not atomically write ", path)
    }
}

safe_name <- function(x) {
    gsub("[^A-Za-z0-9_.-]", "_", x)
}

bfmi <- function(energy) {
    if (length(energy) < 3L || var(energy) == 0) return(NA_real_)
    mean(diff(energy)^2) / var(energy)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (as.character(packageVersion("sigfit")) != "1.2.0") {
    stop("Expected sigfit 1.2.0, found ", packageVersion("sigfit"))
}
if (args$workers < 1L || args$chains < 1L || args$iter < 2L ||
    args$warmup < 0L || args$warmup >= args$iter || args$thin < 1L) {
    stop("Invalid worker or MCMC settings")
}

dataset_filter <- toupper(args$dataset)
cohort_filter <- args$cohort
datasets <- c("TCGA", "CPTAC")
cohorts <- c("Brca", "Ccrcc", "Hnscc", "Lscc", "Luad", "Ov", "Ucec")
if (!dataset_filter %in% c("ALL", datasets)) stop("Invalid dataset filter")
if (!cohort_filter %in% c("ALL", cohorts)) {
    stop("Invalid cohort filter")
}

reference_source_sha256 <- "3762efd072f04bbdbee903417b749ce8fc63378dd4d9042cdc888f155d6815f8"
reference_source_path <- file.path(
    args$analysis_root, "source", "sigfit-v1.2.0", "data", "cosmic_signatures.RData"
)
if (!file.exists(reference_source_path)) stop("Missing vendored COSMIC reference: ", reference_source_path)
hash_output <- system2("sha256sum", reference_source_path, stdout = TRUE, stderr = TRUE)
observed_reference_sha256 <- strsplit(hash_output[[1]], "[[:space:]]+")[[1]][[1]]
if (!identical(observed_reference_sha256, reference_source_sha256)) {
    stop("Vendored COSMIC reference hash mismatch")
}
run_fingerprint <- paste(
    "sigfit-samplewise-v2",
    "sigfit=1.2.0",
    "model=nmf",
    "prior=dirichlet1x30",
    paste0("chains=", args$chains),
    paste0("iter=", args$iter),
    paste0("warmup=", args$warmup),
    paste0("thin=", args$thin),
    paste0("base_seed=", args$base_seed),
    paste0("cosmic_rdata_sha256=", reference_source_sha256),
    "normalization=human-genome-to-emu_then_human-exome-to-nmf",
    sep = ";"
)

data("cosmic_signatures", package = "sigfit")
normalised_signatures <- convert_signatures(
    cosmic_signatures,
    ref_opportunities = "human-genome",
    model_to = "emu"
)
exome_signatures <- convert_signatures(
    normalised_signatures,
    ref_opportunities = "human-exome",
    model_to = "nmf"
)
rownames(exome_signatures) <- paste0("SBS", seq_len(30L))
if (nrow(exome_signatures) != 30L || ncol(exome_signatures) != 96L ||
    max(abs(rowSums(exome_signatures) - 1)) > 1e-12) {
    stop("Invalid exome-normalised COSMIC reference")
}

tasks <- list()
expected <- list()
task_index <- 0L
for (dataset in datasets) {
    for (cohort in cohorts) {
        input_path <- file.path(args$analysis_root, "inputs", dataset, cohort, "sbs96_counts.tsv")
        if (!file.exists(input_path)) stop("Missing input: ", input_path)
        input <- read.delim(input_path, check.names = FALSE, stringsAsFactors = FALSE)
        metadata_columns <- c("Sample", "PATIENT", "snv_count")
        if (!identical(names(input)[seq_along(metadata_columns)], metadata_columns)) {
            stop("Unexpected metadata columns in ", input_path)
        }
        context_columns <- setdiff(names(input), metadata_columns)
        if (length(context_columns) != 96L || anyDuplicated(context_columns)) {
            stop("Expected exactly 96 contexts in ", input_path)
        }
        if (!setequal(context_columns, colnames(exome_signatures))) {
            stop("Input contexts do not match Sigfit COSMIC contexts in ", input_path)
        }
        if (anyDuplicated(input$Sample) || anyDuplicated(input$PATIENT)) {
            stop("Duplicate sample or patient in ", input_path)
        }
        counts <- as.matrix(input[, colnames(exome_signatures), drop = FALSE])
        storage.mode(counts) <- "integer"
        if (any(counts < 0L) || any(rowSums(counts) < 30L) ||
            !identical(as.integer(rowSums(counts)), as.integer(input$snv_count))) {
            stop("Invalid counts in ", input_path)
        }
        output_dir <- file.path(args$analysis_root, "fits", dataset, cohort)
        filenames <- paste0(safe_name(input$Sample), ".tsv")
        if (anyDuplicated(filenames)) stop("Output filename collision in ", input_path)
        expected[[paste(dataset, cohort, sep = "/")]] <- data.frame(
            Sample = input$Sample,
            PATIENT = input$PATIENT,
            snv_count = as.integer(input$snv_count),
            result_path = file.path(output_dir, "samples", filenames),
            stringsAsFactors = FALSE
        )
        for (row in seq_len(nrow(input))) {
            task_index <- task_index + 1L
            tasks[[task_index]] <- list(
                index = task_index,
                dataset = dataset,
                cohort = cohort,
                sample = input$Sample[[row]],
                patient = input$PATIENT[[row]],
                snv_count = as.integer(input$snv_count[[row]]),
                counts = counts[row, , drop = FALSE],
                counts_fingerprint = paste(as.integer(counts[row, ]), collapse = ","),
                # Canonical global task order gives every sample a unique seed,
                # independent of later dataset/cohort filters.
                seed = as.integer(args$base_seed + task_index),
                output_dir = output_dir,
                result_path = file.path(output_dir, "samples", filenames[[row]])
            )
        }
    }
}

if (anyDuplicated(vapply(tasks, function(task) task$seed, integer(1)))) {
    stop("Internal error: sample seeds are not unique")
}
keep_task <- vapply(tasks, function(task) {
    (dataset_filter == "ALL" || task$dataset == dataset_filter) &&
        (cohort_filter == "ALL" || task$cohort == cohort_filter)
}, logical(1))
tasks <- tasks[keep_task]
keep_expected <- vapply(strsplit(names(expected), "/", fixed = TRUE), function(parts) {
    (dataset_filter == "ALL" || parts[[1]] == dataset_filter) &&
        (cohort_filter == "ALL" || parts[[2]] == cohort_filter)
}, logical(1))
expected <- expected[keep_expected]
if (args$max_tasks > 0L) tasks <- tasks[seq_len(min(args$max_tasks, length(tasks)))]

cache_valid <- function(task) {
    if (!file.exists(task$result_path)) return(FALSE)
    cached <- tryCatch(
        read.delim(task$result_path, check.names = FALSE, stringsAsFactors = FALSE, nrows = 1L),
        error = function(error) NULL
    )
    if (is.null(cached) || nrow(cached) != 1L) return(FALSE)
    required <- c(
        "Sample", "PATIENT", "seed", "run_fingerprint", "counts_fingerprint",
        paste0("SBS", seq_len(30L))
    )
    if (any(!required %in% names(cached))) return(FALSE)
    cached_sbs <- as.numeric(unlist(
        cached[1, paste0("SBS", seq_len(30L)), drop = FALSE],
        use.names = FALSE
    ))
    identical(as.character(cached$Sample[[1]]), as.character(task$sample)) &&
        identical(as.character(cached$PATIENT[[1]]), as.character(task$patient)) &&
        identical(as.integer(cached$seed[[1]]), as.integer(task$seed)) &&
        identical(as.character(cached$run_fingerprint[[1]]), run_fingerprint) &&
        identical(as.character(cached$counts_fingerprint[[1]]), task$counts_fingerprint) &&
        all(is.finite(cached_sbs)) &&
        abs(sum(cached_sbs) - 1) <= 1e-8
}

pending <- vapply(tasks, function(task) !cache_valid(task), logical(1))
tasks_to_run <- tasks[pending]
cat("[INFO] total tasks:", length(tasks), "pending:", length(tasks_to_run), "workers:", args$workers, "\n")

fit_one <- function(task) {
    started <- proc.time()[["elapsed"]]
    error_path <- paste0(task$result_path, ".error.txt")
    tryCatch({
        fit <- fit_signatures(
            counts = task$counts,
            signatures = exome_signatures,
            exp_prior = rep(1, 30L),
            model = "nmf",
            chains = args$chains,
            iter = args$iter,
            warmup = args$warmup,
            thin = args$thin,
            seed = task$seed,
            refresh = 0
        )
        exposure_summary <- retrieve_pars(fit, par = "exposures", hpd_prob = 0.95)
        means <- as.numeric(exposure_summary[[1]][1, ])
        lowers <- as.numeric(exposure_summary[[2]][1, ])
        uppers <- as.numeric(exposure_summary[[3]][1, ])
        if (length(means) != 30L || any(!is.finite(means)) || any(means < 0) ||
            any(means > 1) || abs(sum(means) - 1) > 1e-8) {
            stop("Invalid posterior exposure summary")
        }
        stan_summary <- rstan::summary(fit$result, pars = "exposures")$summary
        sampler <- rstan::get_sampler_params(fit$result, inc_warmup = FALSE)
        divergences <- sum(vapply(sampler, function(x) sum(x[, "divergent__"]), numeric(1)))
        max_treedepth <- max(vapply(sampler, function(x) max(x[, "treedepth__"]), numeric(1)))
        treedepth_hits <- sum(vapply(sampler, function(x) sum(x[, "treedepth__"] >= 10), numeric(1)))
        chain_bfmi <- vapply(sampler, function(x) bfmi(x[, "energy__"]), numeric(1))
        row <- data.frame(
            dataset = task$dataset,
            cohort = task$cohort,
            Sample = task$sample,
            PATIENT = task$patient,
            snv_count = task$snv_count,
            seed = task$seed,
            run_fingerprint = run_fingerprint,
            reference_source_sha256 = reference_source_sha256,
            counts_fingerprint = task$counts_fingerprint,
            elapsed_seconds = proc.time()[["elapsed"]] - started,
            max_Rhat = max(stan_summary[, "Rhat"], na.rm = TRUE),
            min_n_eff = min(stan_summary[, "n_eff"], na.rm = TRUE),
            divergences = as.integer(divergences),
            max_treedepth = as.integer(max_treedepth),
            treedepth_hits = as.integer(treedepth_hits),
            min_BFMI = min(chain_bfmi, na.rm = TRUE),
            stringsAsFactors = FALSE
        )
        for (i in seq_len(30L)) row[[paste0("SBS", i)]] <- means[[i]]
        for (i in seq_len(30L)) row[[paste0("SBS", i, "_hpd95_low")]] <- lowers[[i]]
        for (i in seq_len(30L)) row[[paste0("SBS", i, "_hpd95_high")]] <- uppers[[i]]
        atomic_write_table(row, task$result_path)
        if (file.exists(error_path)) unlink(error_path)
        list(ok = TRUE, index = task$index, elapsed = row$elapsed_seconds)
    }, error = function(error) {
        dir.create(dirname(error_path), recursive = TRUE, showWarnings = FALSE)
        writeLines(c(conditionMessage(error), capture.output(traceback())), error_path)
        list(ok = FALSE, index = task$index, error = conditionMessage(error))
    })
}

if (length(tasks_to_run) > 0L) {
    worker_n <- min(args$workers, length(tasks_to_run))
    shard_ids <- (seq_along(tasks_to_run) - 1L) %% worker_n + 1L
    shards <- split(tasks_to_run, shard_ids)
    worker_logs <- file.path(args$analysis_root, "logs", paste0("sigfit_worker_", seq_len(worker_n), ".log"))
    dir.create(dirname(worker_logs[[1]]), recursive = TRUE, showWarnings = FALSE)
    run_shard <- function(shard_id) {
        options(mc.cores = 1L)
        stdout_connection <- file(worker_logs[[shard_id]], open = "at")
        message_connection <- file(worker_logs[[shard_id]], open = "at")
        sink(stdout_connection)
        sink(message_connection, type = "message")
        on.exit({
            sink(type = "message")
            sink()
            close(message_connection)
            close(stdout_connection)
        }, add = TRUE)
        cat("[START] shard", shard_id, "tasks", length(shards[[shard_id]]), "\n")
        results <- lapply(shards[[shard_id]], fit_one)
        cat("[DONE] shard", shard_id, "ok", sum(vapply(results, function(x) x$ok, logical(1))), "\n")
        results
    }
    shard_results <- parallel::mclapply(seq_len(worker_n), run_shard, mc.cores = worker_n)
    flat_results <- unlist(shard_results, recursive = FALSE)
    failed <- flat_results[!vapply(flat_results, function(x) isTRUE(x$ok), logical(1))]
    if (length(failed) > 0L) {
        cat("[FAIL]", length(failed), "sample fits failed\n")
        for (failure in head(failed, 20L)) cat(" task", failure$index, failure$error, "\n")
        quit(save = "no", status = 2L)
    }
}

if (args$max_tasks == 0L) {
    signature_columns <- paste0("SBS", seq_len(30L))
    low_columns <- paste0(signature_columns, "_hpd95_low")
    high_columns <- paste0(signature_columns, "_hpd95_high")
    diagnostic_columns <- c(
        "dataset", "cohort", "Sample", "PATIENT", "snv_count", "seed",
        "run_fingerprint", "reference_source_sha256", "counts_fingerprint",
        "elapsed_seconds", "max_Rhat", "min_n_eff", "divergences",
        "max_treedepth", "treedepth_hits", "min_BFMI"
    )
    all_diagnostics <- list()
    diagnostic_index <- 0L
    for (key in names(expected)) {
        cohort_expected <- expected[[key]]
        missing <- cohort_expected$result_path[!file.exists(cohort_expected$result_path)]
        if (length(missing) > 0L) stop("Missing ", length(missing), " fits for ", key)
        rows <- lapply(cohort_expected$result_path, function(path) {
            read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
        })
        fitted <- do.call(rbind, rows)
        fitted <- fitted[match(cohort_expected$Sample, fitted$Sample), , drop = FALSE]
        if (any(is.na(fitted$Sample)) || anyDuplicated(fitted$Sample) ||
            !identical(as.character(fitted$PATIENT), as.character(cohort_expected$PATIENT))) {
            stop("Aggregated fit identity mismatch for ", key)
        }
        if (max(abs(rowSums(fitted[, signature_columns, drop = FALSE]) - 1)) > 1e-8) {
            stop("Aggregated exposure sum mismatch for ", key)
        }
        parts <- strsplit(key, "/", fixed = TRUE)[[1]]
        cohort_dir <- file.path(args$analysis_root, "fits", parts[[1]], parts[[2]])
        atomic_write_table(fitted[, c("Sample", "PATIENT", "snv_count", signature_columns)],
                           file.path(cohort_dir, "sample_weights.tsv"))
        atomic_write_table(fitted[, c("PATIENT", signature_columns)],
                           file.path(cohort_dir, "patient_weights.tsv"))
        atomic_write_table(fitted[, c("Sample", "PATIENT", "snv_count", low_columns)],
                           file.path(cohort_dir, "sample_weights_hpd95_lower.tsv"))
        atomic_write_table(fitted[, c("Sample", "PATIENT", "snv_count", high_columns)],
                           file.path(cohort_dir, "sample_weights_hpd95_upper.tsv"))
        atomic_write_table(fitted[, diagnostic_columns],
                           file.path(cohort_dir, "sample_diagnostics.tsv"))
        diagnostic_index <- diagnostic_index + 1L
        all_diagnostics[[diagnostic_index]] <- fitted[, diagnostic_columns]
    }
    combined_diagnostics <- do.call(rbind, all_diagnostics)
    atomic_write_table(combined_diagnostics, file.path(args$analysis_root, "fits", "all_sample_diagnostics.tsv"))
    metadata <- data.frame(
        sigfit_version = as.character(packageVersion("sigfit")),
        model = "nmf",
        signature_reference = "sigfit 1.2.0 COSMIC v2 SBS1-SBS30, exome-normalised",
        exposure_prior = "Dirichlet(1 x 30)",
        computational_factorization = "independent per-sample fits of the factorized fixed-signature posterior",
        chains = args$chains,
        iter = args$iter,
        warmup = args$warmup,
        thin = args$thin,
        base_seed = args$base_seed,
        run_fingerprint = run_fingerprint,
        reference_source_sha256 = reference_source_sha256,
        sample_n = nrow(combined_diagnostics),
        max_Rhat = max(combined_diagnostics$max_Rhat, na.rm = TRUE),
        total_divergences = sum(combined_diagnostics$divergences),
        total_treedepth_hits = sum(combined_diagnostics$treedepth_hits),
        stringsAsFactors = FALSE
    )
    atomic_write_table(metadata, file.path(args$analysis_root, "fits", "fit_metadata.tsv"))
}

cat("[OK] Sigfit sample-wise fitting complete\n")
