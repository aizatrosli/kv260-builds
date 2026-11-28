# Builds the optional C host kernels (tpurt/_lib/libtpurt_host-<arch>.so; tpurt builds it on first use).  Works natively on the KV260
# (aarch64 gcc) and on x86.  Cross: make CC=aarch64-linux-gnu-gcc
CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a53
endif
LIB := tpurt/_lib/libtpurt_host-$(ARCH).so

all: $(LIB)

$(LIB): csrc/tpurt_host.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

test:
	python3 -m pytest -q tests

clean:
	rm -rf tpurt/_lib

.PHONY: all test clean
