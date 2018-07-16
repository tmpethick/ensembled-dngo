build:
	docker build -t robo .

run:
	docker kill robo1 || true
	docker stop robo1 || true
	docker rm robo1 || true
	docker run -d -it --name robo1 -p 8888:8888 -v "$$(pwd)":/root/src/mount robo
	docker logs robo1

stop:
	docker stop robo1

start:
	docker start robo1

clear:
	docker stop robo1
	docker rm robo1 || true
	docker rmi robo
