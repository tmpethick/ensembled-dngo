build:
	docker build -t robo .

run: removecontainer
	docker run -d -it --name robo1 -p 8888:8888 -v "$$(pwd)/shared":/root/shared robo jupyter lab --ip=0.0.0.0 --no-browser --allow-root /root/shared
	@echo $(docker logs robo1)

stop:
	docker stop robo1

start:
	docker start robo1

removecontainer:
	docker kill robo1 || true
	docker stop robo1 || true
	docker rm robo1 || true

clear: removecontainer
	docker rmi robo
