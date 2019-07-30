#!/bin/bash - 
#===============================================================================
#
#          FILE: make_interps.sh
# 
#         USAGE: ./make_interps.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 07/05/2019 19:06
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
d=`pwd`
for f in ./interps/*/*
do
  cd $f
  ffmpeg -y -framerate 16 -pattern_type glob -i "*.png"  -vcodec libx264 -acodec aac out-256.mp4
  cd $d
done

