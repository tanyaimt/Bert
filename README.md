# Bert
in Ubuntu WSL setup following the next link
https://docs.microsoft.com/en-us/windows/wsl/install
Initial Setup
Activate virtual enviroment 
1) sudo apt install python3-virtualenv
2) virtualenv -p python3 direnv
3) source direnv/bin/activate;clear #validate the (direnv) user@hostnme:~/path/Bert$
To exit the virtual enviroment
4) deactivate 

Install Bert
!pip install bert-for-tf2
!pip install sentencepiece
python -m pip install pandas
python -m pip install bs4
pip install --upgrade tensorflow
pip install --upgrade tensorflow_hub



Create Personal Access Token on GitHub
https://stackoverflow.com/questions/68775869/support-for-password-authentication-was-removed-please-use-a-personal-access-to

From your GitHub account, go to Settings => Developer Settings => Personal Access Token => Generate New Token (Give your password) => Fillup the form => click Generate token => Copy the generated Token, it will be something like ghp_sFhFsSHhTzMDreGRLjmks4Tzuzgthdvfsrta

GIT API
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

git config --global user.name "your_github_username"
git config --global user.email "your_github_email"
git config -l

git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
> Cloning into `Spoon-Knife`...
Username for 'https://github.com' : type username
Password for 'https://github.com' : give your personal access token here

git config --global credential.helper cache

If needed, anytime you can delete the cache record by:
$ git config --global --unset credential.helper
$ git config --system --unset credential.helper