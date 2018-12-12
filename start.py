import os;


print("start to run services...");
print("start nginx----------------------------------------------------------");
os.system('nginx');

print("start mailcatcher----------------------------------------------------")
os.system('mailcatcher')

print("start activemq-------------------------------------------------------")
os.system('/home/dylanduan/codesoft/apache-activemq-5.10.1/bin/activemq start')

print("start redis----------------------------------------------------------")
os.system('redis-server')


os.system('redis-cli ping');
