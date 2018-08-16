TEST_PATH=./tests
W=4:00

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

movemnist:
	rsync -av ./raw pethickt@euler.ethz.ch:~/bonn
	rsync -av ./processed pethickt@euler.ethz.ch:~/bonn

run:
	rsync -avu ./src pethickt@euler.ethz.ch:~/bonn
	rsync -av ./run.py pethickt@euler.ethz.ch:~/bonn
	rsync -av ./config.py pethickt@euler.ethz.ch:~/bonn
	rsync -av ./hpc.sh pethickt@euler.ethz.ch:~/bonn
	ssh pethickt@euler.ethz.ch 'cd $$HOME/bonn && bsub -W ${W} sh hpc.sh ${ARGS}'

pull:
	# Copy all folders in outputs/
	rsync -avu --include='*/' --exclude='/*' pethickt@euler.ethz.ch:~/bonn/outputs/ ./outputs
	rsync -av pethickt@euler.ethz.ch:~/bonn/outputs/entries.csv ./outputs/entries.remote.csv
	python database_merger.py
	rm -f outputs/entries.remote.csv

.PHONY: test
