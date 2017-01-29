#!/bin/bash

CARND_IP=54.152.23.19
echo $CARND_ID

python ./generate_model.py
scp -i ~/.ssh/carnd_aws ./bc_utils.py carnd@$CARND_IP:/home/carnd/behavioral-cloning
scp -i ~/.ssh/carnd_aws ./generate_model.py carnd@$CARND_IP:/home/carnd/behavioral-cloning
scp -i ~/.ssh/carnd_aws ./model.py carnd@$CARND_IP:/home/carnd/behavioral-cloning
ssh -i ~/.ssh/carnd_aws carnd@$CARND_IP 'source /home/carnd/anaconda3/bin/activate carnd-term1 && cd /home/carnd/behavioral-cloning && python model.py'
scp -i ~/.ssh/carnd_aws carnd@$CARND_IP:/home/carnd/behavioral-cloning/model.h5 ./model.h5
