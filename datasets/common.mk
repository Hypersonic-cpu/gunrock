#The following variables must be defined prior to including this
#makefile fragment
#
#GRAPH_URL:  the url path to the file

OSUPPER := $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

ifeq (DARWIN, $(findstring DARWIN, $(OSUPPER)))
    WGET ?= curl -O
else
    WGET ?= wget -N
endif

# Command used to download GRAPH_URL. The command must accept the URL as its
# final argument and save GRAPH_FILE in the current directory. Override it on
# the make command line, for example:
#   make -C datasets/roadNet-CA \
#     DOWNLOAD_CMD='curl --socks5-hostname localhost:7888 --location --fail --remote-name'
DOWNLOAD_CMD ?= $(WGET)
DOWNLOAD_RETRIES ?= 5

TAR  := tar
GZIP := gzip
MATRIX2SNAP := ../matrix2snap.py

GRAPH_FILE := $(notdir $(GRAPH_URL))

all: setup

fetch: $(GRAPH_FILE)

$(GRAPH_FILE):
	@attempt=1; \
	while [ $$attempt -le $(DOWNLOAD_RETRIES) ]; do \
	  echo "Download attempt $$attempt/$(DOWNLOAD_RETRIES): $(GRAPH_URL)"; \
	  if $(DOWNLOAD_CMD) $(GRAPH_URL); then \
	    exit 0; \
	  fi; \
	  attempt=$$((attempt + 1)); \
	done; \
	echo "Download failed after $(DOWNLOAD_RETRIES) attempts" >&2; \
	exit 1

IPDPS17: setup
