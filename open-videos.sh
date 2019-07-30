#!/bin/bash - 
#===============================================================================
#
#          FILE: open-videos.sh
# 
#         USAGE: ./open-videos.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 07/14/2019 02:21
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
mpv ./final-results/random.mp4 --loop &
mpv ./final-results/burger-burger-cycle.mp4 --loop &
mpv ./final-results/pizza-pizza-cycle.mp4 --loop &
mpv ./final-results/pizza-burger-cycle.mp4 --loop &

