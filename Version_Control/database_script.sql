CREATE DATABASE IF NOT EXISTS chessdata;
CREATE USER IF NOT EXISTS 'ant'@'localhost' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON `chessdata`.* TO 'ant'@'localhost';

-- cat setup_mysql_dev.sql | mysql -hlocalhost -uroot -p