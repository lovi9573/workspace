#!/bin/sh

MODEL_DIR="\/home\/jlovitt\/storage\/ILSVRC2012\/models"
DATA_DIR="\/home\/jlovitt\/storage\/ILSVRC2012"



RE2="s/\$DATA_DIR/$DATA_DIR/g"

for DIR in $MODEL_DIR/*/; do
	for FILE in $DIR*.p; do
		echo "$FILE"
		sed  "s%\$MODEL_DIR%$DIR%g" <$FILE | sed  $RE2 >"$FILE"rototxt
	done
done
