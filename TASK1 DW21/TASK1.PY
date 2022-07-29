import re
import pandas as pd
from pip import main
import sys


#Function to Validate User Name 
def Validate_USNAME(name):
  regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
  if re.fullmatch(regex, name):
    return True
  else:
    return False


#Function to Validate User Password 
def Validate_USPASS(password):
  regex = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&_-])[A-Za-z\d@$!#%*?&_-]{5,16}$')
  if re.fullmatch(regex, password):
   return True
  else:
    return False

#Function to Read and Write to CSV File. CSV File name: " MAINDATA1.CSV"
def MainData(opt,name,password):
  df= pd.read_csv("MAINDATA1.csv")
  if opt==1:         #CHK IF USER NAME EXIST
    x=df.index[df['EMAIL']==name].tolist()
    if x==[]:
      return False
    else:
      return True

  if opt==2:        #GET USER PASSWORD FOR EXIST USER
    x2=df.index[df['EMAIL']==name].tolist()
    y1=int(*x2)
    PassQ=df.loc[y1,'PASSWORD']
    return PassQ
  if opt==3:        #UPDATE PASSWORD FOR EXIST USER
    x3=df.index[df['EMAIL']==name].tolist()
    df.loc[x3,'PASSWORD']=password
  if opt==4:        #APPEND NEW USER
    dfappend = pd.DataFrame({"EMAIL":[name],"PASSWORD":[password]})
    dfappend.to_csv('MAINDATA1.csv', mode='a', index=False, header=False)  


#Function to find User name from CSV File
def FindUS_name(name):
  FindUS=MainData(1,name,0)
  if FindUS==True:
    return True
  else:
    return False

#Function to find User Password from CSV File
def FindUS_Pass(name):
  Pass=MainData(2,name,0)
  return Pass

#Function to Reset User Password
def Pass_Reset():
  Temp_USname=input("Enter Username: ")
  Test_PassRest=FindUS_name(Temp_USname)
  exit_chk=False
  if Test_PassRest==True:
    Set8=False
    while(Set8!=True):
      Temp_Pass=input("Enter New Password: Password mush be 5-16 Characters, It must have minimum one special character, one digit, one uppercase, and one lowercase character\n ")
      Set8=Validate_USPASS(Temp_Pass)
      if Set8==False:
        print("Password not Valid")
      else:       
        MainData(2,Temp_USname,Temp_Pass)
        
    print("Password Reset Sucesfull")
    main_screen()
  else:
    Set3=False
    print("Username Not Found\n")
    while(Set3!=True):
      try:
        Option_1=int(input("Enter --> 1 to Register as New User\n Enter --> 2 to Exit \n "))
      except ValueError:
          print("Enter Valid Numeric Option Only")
          continue
      if Option_1==1:
          User_REG()
          exit_chk=True
          Set3=True
      elif Option_1==2:
        exit_chk=True
        break
  if exit_chk==True:
    main_screen()
  else:
    return False


#Function to find User Password from CSV File
def CheckUS_Pass(US_name):
  Set=False
  Test=FindUS_name(US_name)
  if Test==True:
    US_Pass=FindUS_Pass(US_name)
    
    while(Set!=True):
      US_GETPASS=input("Enter Password:  ")
      if US_Pass==US_GETPASS:
        print("Login Sucessful")
        Set=True
      else:
        print("Password Incorrect")
        Set2=False
        while(Set2!=True):
          try:
            Option2=int(input("  Press--> 1 to Retry \n  Press --> 2 to Reset Password \n"))
          except ValueError:
            print("Enter Valid Numeric Option Only")
            continue
          if Option2==1:
            Set2=True
          elif Option2==2:
            Pass_Reset()
            Set=True
            Set2=True
            break
          elif Option2 not in[1,2]:
            print("enter Valid Option")
  else:
    print("User Not Found")
    main_screen()
  return True

#Function for USer Registration
def User_REG():
  Set6=False
  while(Set6!=True):
    US_NEW_NAME=input("Enter Valid Email:  ")
    Chk_US_NAME=Validate_USNAME(US_NEW_NAME)
    if Chk_US_NAME==True:
      Set6=True
  
  Set7=False
  while(Set7!=True):    
      US_NEW_PASS=input("Enter Valid Password, Password mush be 5-16 Characters, It must have minimum one special character, one digit, one uppercase, and one lowercase character \n")
      Chk_US_PASS=Validate_USPASS(US_NEW_PASS)
      if Chk_US_PASS==True:
        Set7=True

  MainData(4,US_NEW_NAME,US_NEW_PASS)
  print("Registration Successful ")
  main_screen()



#Function for Welcome Screen
def main_screen():
    Set4=False
    while(Set4!=True):
        try:
          Option2=int(input(" Welcome to Vignesh School\n Select --> 1 for Sign In \n Select --> 2  for New Registration \n Select --> 3 Forgot Password \n Select 4 To Exit School "))
        except ValueError:
          print("Enter Valid Numeric Option Only")
          continue
        if Option2==1:
          User_SIGNIN() 
          Set4=True
        elif Option2==2:
          User_REG()
        elif Option2==3:
          Pass_Reset()
        elif Option2==4:
          print("Thank You Visit Agiain")
          #break
          #quit()
          sys.exit()
          #raise SystemExit
          Set4=True
        else:
          print("Enter Valid Option")
    
def User_SIGNIN():
  US_name=input("Enter Email :  ")
  CheckUS_Pass(US_name)
main_screen()
