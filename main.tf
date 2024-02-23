provider "aws" {
  region = "us-east-1"  # Change this to your desired AWS region
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "crypto-monitoring-app"
  }

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y python3-pip
              git clone https://github.com/yourusername/your-repo.git /home/ubuntu/cryptocurrency-monitoring-app
              cd /home/ubuntu/cryptocurrency-monitoring-app
              pip3 install -r requirements.txt
              python3 app.py
              EOF
}

output "public_ip" {
  value = aws_instance.web.public_ip
}
