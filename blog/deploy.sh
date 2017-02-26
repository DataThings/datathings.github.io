export LC_TIME=en_US.UTF-8
if [ -d "public" ]; then
  cd public
  git pull
  cd ..
else
  git clone git@github.com:datathings/datathings.github.io.git public
fi
hugo -d "./public/blog"
cd public/
git add -A
MESSAGE="Site rebuild $(date)"
git commit -m "$MESSAGE"
git push
