## Instructions for setting up AWS instance

This instructions are only for people with a CIS-aws account

1. Install AWS Command Line Interface.
  (a.) `pip install awscli`
    Note: might need `--ignore-installed six` flag for El Capitan

2. Configure AWS Command Line Interface

  (a.) Generate Access Key and Secret Access ID
      i. go sign in at https://brown-cis.signin.aws.amazon.com/console
     ii. Go to Services >> IAM >> Users
    iii. Click on your username
     iv. Click on Security Credentials tab
      v. Click Create access key (then download .csv)

  (b.) run the following from terminal: aws configure
      i. you'll be prompted to enter: Access Key ID, Secret Access Key, default region (us-east-1), and default output format (I suggest json or text)

3. Launch EC2 Instance with shell script
  (a.) bash aws_instance_init.sh

4. Stop (and/or Terminate) an instance:
  (a.) bash stop_instance.sh
  (b.) bash terminate_instance.sh

5. Connect
ssh -i ~/.ssh/your-key.pem ec2-user@‘public DNS from console’

### Notes:
 - `jupyter notebook` seems to only give access to Python 2
 - `ipython notebook` gives access to Python 3

Got to
https://[public ip from console]:8888/
