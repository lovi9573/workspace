#!/bin/sh

C7H=${HOME}/.local/lib/cudnn-7.0/cudnn.h
C6H=${HOME}/.local/lib/cudnn-6.5/cudnn.h
LNH=${HOME}/.local/include/cudnn.h
C7L=${HOME}/.local/lib/cudnn-7.0/libcudnn.so
C6L=${HOME}/.local/lib/cudnn-6.5/libcudnn.so
LNL=${HOME}/.local/lib/libcudnn.so

rm -f ${LNH} ${LNL}

if [ $1 == "7" ]; then
	ln -s ${C7L} ${LNL}
	ln -s ${C7H} ${LNH};
fi

if [ $1 == "6" ]; then
	ln -s ${C6L} ${LNL}
	ln -s ${C6H} ${LNH};
fi

