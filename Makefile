TEST_PATH=./tests
W=4:00

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

pushdata:
	rsync -av ./raw pethickt@euler.ethz.ch:~/bonn
	rsync -av ./processed pethickt@euler.ethz.ch:~/bonn
	rsync -av ~/.hpolib pethickt@euler.ethz.ch:~/

pushdata-leonhard:
	rsync -av ./raw pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ./processed pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ~/.hpolib pethickt@login.leonhard.ethz.ch:~/

push:
	rsync -avu ./src pethickt@euler.ethz.ch:~/bonn
	rsync -av ./run.py pethickt@euler.ethz.ch:~/bonn
	rsync -av ./config.py pethickt@euler.ethz.ch:~/bonn
	rsync -av ./hpc.sh pethickt@euler.ethz.ch:~/bonn
	rsync -av ./hpc_euler.sh pethickt@euler.ethz.ch:~/bonn

push-leonhard:
	rsync -avu ./src pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ./run.py pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ./config.py pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ./hpc.sh pethickt@login.leonhard.ethz.ch:~/bonn
	rsync -av ./hpc_leonhard.sh pethickt@login.leonhard.ethz.ch:~/bonn

run:
	ssh pethickt@euler.ethz.ch 'cd $$HOME/bonn && bsub -W ${W} sh hpc_euler.sh ${ARGS}'

run-leonhard:
	ssh pethickt@login.leonhard.ethz.ch 'cd $$HOME/bonn && bsub -R "rusage[mem=8000,ngpus_excl_p=1]" -W ${W} sh hpc_leonhard.sh ${ARGS}'

pull:
	# Copy all folders in outputs/
	rsync -avu --include='*/' --exclude='/*' pethickt@euler.ethz.ch:~/bonn/outputs/ ./outputs
	rsync -av pethickt@euler.ethz.ch:~/bonn/outputs/entries.csv ./outputs/entries.remote.csv
	python database_merger.py
	rm -f outputs/entries.remote.csv

pull-leonhard:
	# Copy all folders in outputs/
	rsync -avu --include='*/' --exclude='/*' pethickt@login.leonhard.ethz.ch:~/bonn/outputs/ ./outputs
	rsync -av pethickt@login.leonhard.ethz.ch:~/bonn/outputs/entries.csv ./outputs/entries.remote.csv
	python database_merger.py
	rm -f outputs/entries.remote.csv

.PHONY: test
