docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker run --name mysql-test -v ./mysql/db:/etc/mysql/conf.d -v ./mysql/db:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=1234 -p 3308:3308 -d mysql:latest
