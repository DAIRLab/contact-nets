# Need runusers because root doesn't have python3.7
echo "display" >> /home/samuel/startup.txt 
runuser -l samuel -c 'Xvfb :5 -screen 0 800x600x24 &'

if test -f "/home/samuel/SoPhTER/out/trainer.pt"; then
    echo "restarting" >> /home/samuel/startup.txt
    runuser -l samuel -c 'export PYTHONPATH=/home/samuel/SoPhTER; export DISPLAY=:5; python3.7 /home/samuel/SoPhTER/contactnets/experiments/{experiment}/data_sweep.py --resume {sweep_options} &> /home/samuel/train_resume.txt'
else
    echo "pulling" >> /home/samuel/startup.txt
    runuser -l samuel -c 'git -C /home/samuel/SoPhTER pull'
    while [ $? -ne 0 ]; do
        echo "pulling again" >> /home/samuel/startup.txt
        runuser -l samuel -c 'git -C /home/samuel/SoPhTER pull'
    done

    echo "checking out" >> /home/samuel/startup.txt
    runuser -l samuel -c 'git -C /home/samuel/SoPhTER checkout {hash}'

    echo "saving git stuff" >> /home/samuel/startup.txt
    runuser -l samuel -c 'git -C /home/samuel/SoPhTER log --name-status HEAD^..HEAD >> /home/samuel/SoPhTER/out/git.txt'


    echo "sweep" >> /home/samuel/startup.txt
    runuser -l samuel -c 'export PYTHONPATH=/home/samuel/SoPhTER; export DISPLAY=:5; python3.7 /home/samuel/SoPhTER/contactnets/experiments/{experiment}/data_sweep.py {sweep_options} &> /home/samuel/train.txt'
fi
