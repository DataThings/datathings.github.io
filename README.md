# DataThings blog

## Requirements

Install HUGO blog engine, available at `https://gohugo.io`
On mac, simply do a `brew install hugo`

## Add a post

- git clone the blog branch
- create the new post file
  ```sh
  hugo new post/my-great-post.md
  ```
- edit the newly created files
  - if any static content use the static directory (images...)
- sync the master branch:
```sh
sh deploy.sh
```
