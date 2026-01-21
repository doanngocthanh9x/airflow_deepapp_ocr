-- Create database
CREATE DATABASE IF NOT EXISTS airflow_db;

-- Create user and grant privileges
CREATE USER IF NOT EXISTS 'airflow_user'@'%' IDENTIFIED BY 'airflow_pass123';
GRANT ALL PRIVILEGES ON airflow_db.* TO 'airflow_user'@'%';
GRANT ALL PRIVILEGES ON *.* TO 'airflow_user'@'%';

-- Flush privileges
FLUSH PRIVILEGES;
