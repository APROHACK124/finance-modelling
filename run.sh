sudo docker stop myapp
sudo docker rm myapp
sudo docker build -t app-build .
sudo docker run -v $(pwd)/data:/data -d -p 8080:5000 --name myapp app-build
