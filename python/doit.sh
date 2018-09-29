sequential tRexTraining(qsub="-hard -l h_vmem=15G -l h_rt=80:00:00"):
	curPath=$(pwd)
	pythonExec='/u/platen/virtualenvironment/tRex/bin/python'
	echo "$(ls /usr/bin/google-chrome)"
	tRexExec=${curPath}/main.py
	${pythonExec} ${tRexExec} 
