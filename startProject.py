import os

os.chdir("/home/dylanduan/projects/anxstatic-server")
os.system("pm2 start project.json")


os.chdir("/home/dylanduan/projects/megaidea/push/push")
os.system("pm2 start push.json")


os.chdir("/home/dylanduan/projects/anxconsole-server")
os.system("pm2 start project.json")
