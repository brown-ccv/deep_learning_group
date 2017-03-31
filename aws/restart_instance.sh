#!/bin/bash
aws ec2 start-instances --instance-ids i-0c0f5839b3eafb450 --output table
echo Waiting for instance start...
aws ec2 wait instance-running --instance-ids i-0c0f5839b3eafb450
sleep 90
export instanceUrl=`aws ec2 describe-instances --instance-ids i-0c0f5839b3eafb450 --query Reservations[0].Instances[0].PublicDnsName --output text`
echo "Your instance URL is: $instanceUrl"
