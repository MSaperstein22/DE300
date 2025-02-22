docker run -v /Users/mattsaperstein/DE300/Lab6/:/tmp/wc-demo -it \
       -p 8888:8888 \
           --name wc-container \
       pyspark-image