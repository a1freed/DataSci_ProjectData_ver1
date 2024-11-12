
# Initialize Git
!git init

# Add your file to the staging area
!git add DataSci_ProjectData_ver1.py

# Commit your changes
!git commit -m "Initial commit"

# Add your GitHub repository as a remote (without angle brackets)
!git remote add origin https://github.com/a1freed/DataSci_ProjectData_ver1.git

# Verify the remote has been added correctly
!git remote -v

# Push to the main branch (or master if that's what you're using)
!git push -u origin master  # or main if that's your branch name
