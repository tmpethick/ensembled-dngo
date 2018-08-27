

- hpc
  https://scicomp.ethz.ch/lsf_submission_line_advisor/
- Docker with notebook
- IBM LSF HPC (bash)
- Bash
  find . -name '*.orig' #-delete
  find . -name 'lsf.*' -delete
  find . -type f -print | xargs grep "example"
- ssh
  cat $HOME/.ssh/secure_rsa.pub | ssh pethickt@login.leonhard.ethz.ch "cat - >> .ssh/authorized_keys"
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys
- Pandas
  https://jeffdelaney.me/blog/useful-snippets-in-pandas/

