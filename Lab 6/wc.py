from pyspark import SparkContext, SparkConf

DATA = "./data/*.txt"
OUTPUT_DIR = "counts" # name of the folder

# Word Count
# Save only the words that have count greater or equal to 3.

def word_count():
    sc = SparkContext("local","Word count example")
    textFile = sc.textFile(DATA)
    counts = textFile.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).filter(lambda pair: pair[1] >= 3)  # Keep only words with count >= 3
    counts.saveAsTextFile(OUTPUT_DIR)
    print("Number of partitions: ", textFile.getNumPartitions())
word_count()
