#!/usr/bin/make
# Makefile for Monte Carlo simulation code.
# Nicholas Dou, 2014

SHELL = /bin/sh

CC = g++

DBFLAGS = -O0 -g -Wall -Wextra -Wpedantic

# clang++
# CPPFLAGS = -stdlib=libc++ -O3
# CPPFLAGS = -stdlib=libc++ $(DBFLAGS)

# g++
# CPPFLAGS = -fopenmp -std=c++98 -O3
CPPFLAGS = -fopenmp -std=c++98 $(DBFLAGS)

INCLUDES =

LDFLAGS = -L/usr/local/lib

LIBS = -lboost_random

SRCS = material.cpp problem.cpp

OBJS = $(SRCS:.cpp=.o)
EXECS = run

.PHONY: all
all: $(EXECS)

run: main.cpp $(OBJS)
	$(CC) $(CPPFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) $(OBJS) $< -o $@

%.o : %.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) $< -o $@

.PHONY: clean
clean:
	-rm $(OBJS) $(EXECS)