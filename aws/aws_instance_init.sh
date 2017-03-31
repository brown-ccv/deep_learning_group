#!/bin/bash
# set -x # Use for debug mode

# settings
export instanceType="p2.xlarge"
export name="$USER-datasci"

hash aws 2>/dev/null
if [ $? -ne 0 ]; then
    echo >&2 "'aws' command line tool required, but	 not installed.  Aborting."
    exit 1
fi

if [ -z "$(aws configure get aws_access_key_id)" ]; then
    echo "AWS credentials not configured... Aborting"
    exit 1
fi

# get the correct ami
export ami="ami-0ad3791c" 					# AWS Deep Learning AMI w/ MXNet, TF, Caffe, Torch

# select our virtual private cloud
export vpcId='vpc-948c14f2'
aws ec2 create-tags --resources $vpcId --tags --tags Key=Name,Value=$name

export subnetId='subnet-0114432c'
export securityGroupId='sg-dcd3b3a0'


if [ ! -d ~/.ssh ]; then
	mkdir ~/.ssh
fi

if [ ! -f ~/.ssh/aws-key-$name.pem ]; then
	aws ec2 create-key-pair --key-name aws-key-$name --query 'KeyMaterial' --output text > ~/.ssh/aws-key-$name.pem
	chmod 400 ~/.ssh/aws-key-$name.pem
fi

export instanceId=`aws ec2 run-instances --image-id $ami --count 1 --instance-type $instanceType --key-name aws-key-$name --security-group-ids $securityGroupId --subnet-id $subnetId --query 'Instances[0].InstanceId' --output text`
aws ec2 create-tags --resources $instanceId --tags --tags Key=Name,Value=$name-gpu-machine

echo Waiting for instance start...
aws ec2 wait instance-running --instance-ids $instanceId
sleep 90 													# wait for ssh service to start running too
export instanceUrl=`aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicDnsName' --output text`

# reboot instance, because I was getting "Failed to initialize NVML: Driver/library version mismatch"
# error when running the nvidia-smi command
# see also http://forums.fast.ai/t/no-cuda-capable-device-is-detected/168/13
aws ec2 reboot-instances --instance-ids $instanceId


echo Connect to your instance: ssh -i ~/.ssh/aws-key-$name.pem ec2-user@$instanceUrl


# Write shell script to stop instance
echo "#!/bin/bash" > stop_instance.sh
echo aws ec2 stop-instances --instance-ids $instanceId 1\>/dev/null >> stop_instance.sh
echo echo "Stoping instance..." >> stop_instance.sh
chmod +x stop_instance.sh


# Write shell script to terminate instance
echo "#!/bin/bash" > terminate_instance.sh
echo echo "Terminating instance..." >> terminate_instance.sh
echo aws ec2 terminate-instances --instance-ids $instanceId 1\>/dev/null >> terminate_instance.sh
echo echo "Deleting stop_instance.sh..." >> terminate_instance.sh
echo rm stop_instance.sh >> terminate_instance.sh
echo echo "Deleting restart_instance.sh..." >> terminate_instance.sh
echo rm restart_instance.sh >> terminate_instance.sh
echo echo "Deleting terminate_instance.sh..." >> terminate_instance.sh
echo rm terminate_instance.sh >> terminate_instance.sh
echo echo "Done! All scripts related to instance $instanceId have been deleted." >> terminate_instance.sh
chmod +x terminate_instance.sh


# Write shell script to restart instance
echo "#!/bin/bash" > restart_instance.sh
echo aws ec2 start-instances --instance-ids $instanceId --output table >> restart_instance.sh
echo echo Waiting for instance start... >> restart_instance.sh
echo aws ec2 wait instance-running --instance-ids $instanceId >> restart_instance.sh
echo sleep 90 >> restart_instance.sh
echo export instanceUrl=\`aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicDnsName' --output text\` >> restart_instance.sh

echo echo \"Your instance URL is: \$instanceUrl\" >> restart_instance.sh
chmod +x restart_instance.sh
