# Server Setup instructions

# Create a ubuntu 22.04 server on digital ocean

- Create a new droplet
- Choose ubuntu 22.04
- choose a plan (i went with the cheapest)
- create a password

# Connect to the server

- Open terminal
- ssh root@your_server_ip
- enter your password

- Can also use VS Code to connect to the server:
  - Install the Remote - SSH extension
  - Click on the green button in the bottom left corner
  - Click on Remote-SSH: Connect to Host
  - Enter the server ip
  - Enter the username
  - Enter the password

# Configure the server

- Update the server
  - sudo apt update
  - sudo apt upgrade
- Install python
  - Get the Needed Software
    `sudo add-apt-repository ppa:deadsnakes/ppa`
    `sudo apt install python3.11`
  - Install pip
    `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py` while in the python directory
    `python3.11 get-pip.py`
  - Check that pip was installed
    `pip --version`

# Clone the project

- `cd var`
- `mkdir www`
- `cd www`
- Clone the repo
  - `git clone https://github.com/sethlenhof/Senior-Design-Mapping-for-the-Masses.git`
  - Install the requirements
  - `cd Senior-Design-Mapping-for-the-Masses`
  - `pip install -r requirements.txt`

# Run the server

- Run this to install libgl dependencies for o3d, `sudo apt install libgl1 libgomp1`

# Test the ability to run the scripts

- `cd scripts`
- `python3.11 backend.py`
- `python3.11 backend2.py`

* **Note I had issues with this initially failing or causing seg faults. I had to restart the droplet after waiting a while. Another thing up for discussion possibly is getting a droplet with more memory, currently I am using the cheapest option: 0.5gb ram, 1 cpu.**

# Installing / Configuring Nginx

- `sudo apt install nginx`

- Configure the Nginx reverse proxy `sudo nano /etc/nginx/sites-available/default`
- Add the following to the file:

  ```
  server {
    listen 80;
    server_name YOUR_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
  }
  ```

  ### Run the following commands:

  - `sudo systemctl restart nginx`
  - `sudo systemctl enable nginx`

# Install / Setup Gunicorn

- **Ensure you are in the root directory of the project and in the venv**
- `pip install gunicorn`
- Test the connection: `gunicorn --workers 3 --bind 0.0.0.0:8000 app:app`
- Using an api tool of your choosing, test the connection to the server by sending a GET request to `http://your_server_ip/getBackendpng` and you should get a return image.

## Create the gunicorn service

- Create a new service file: `sudo nano /etc/systemd/system/gunicorn.service`
- Add the following to the file:

  ```
  [Unit]
  Description=Gunicorn instance to serve Senior-Design-Mapping-for-the-Masses
  After=network.target

  [Service]
  User=www-data
  Group=www-data
  WorkingDirectory=/var/www/Senior-Design-Mapping-for-the-Masses
  ExecStart=/var/www/Senior-Design-Mapping-for-the-Masses/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 app:app

  [Install]
  WantedBy=multi-user.target
  ```

- Start the service: `sudo systemctl start gunicorn`
- Enable the service: `sudo systemctl enable gunicorn`
- Ensure the service is running: `sudo systemctl status gunicorn`
