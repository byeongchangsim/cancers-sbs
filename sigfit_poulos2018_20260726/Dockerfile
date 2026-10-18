FROM rocker/r-ver:3.4.1

ARG CRAN_SNAPSHOT=2018-07-23
ARG SIGFIT_COMMIT=27a9ff5ff4f491476815b532564ae1e7955ed001

ENV CRAN_REPO=https://packagemanager.posit.co/cran/${CRAN_SNAPSHOT}
LABEL org.opencontainers.image.source="https://github.com/kgori/sigfit" \
      org.opencontainers.image.version="1.2.0" \
      org.opencontainers.image.revision="${SIGFIT_COMMIT}" \
      org.opencontainers.image.description="Historical Sigfit 1.2.0 runtime for Poulos et al. 2018"

RUN printf 'options(repos = c(CRAN = "%s"))\n' "${CRAN_REPO}" \
      > /usr/local/lib/R/etc/Rprofile.site \
    && install2.r --error rstan coda ggplot2 clue \
    && Rscript -e 'expected <- c(rstan="2.17.3", StanHeaders="2.17.2", Rcpp="0.12.17"); actual <- vapply(names(expected), function(x) as.character(packageVersion(x)), character(1)); print(actual); stopifnot(identical(unname(actual), unname(expected)))'

COPY source/sigfit-v1.2.0 /opt/sigfit-v1.2.0

RUN cd /opt/sigfit-v1.2.0 \
    && mkdir -p src/include \
    && Rscript tools/make_cpp.R \
    && cat src/*Module.cc > src/Modules.cpp \
    && rm src/*Module.cc \
    && R CMD INSTALL --no-multiarch --with-keep.source . \
    && Rscript -e 'stopifnot(as.character(packageVersion("sigfit")) == "1.2.0"); library(sigfit)'

WORKDIR /work
