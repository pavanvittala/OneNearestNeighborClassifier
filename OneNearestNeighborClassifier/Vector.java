import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.PriorityQueue;

public class Vector {
	private int documentCount;	//Total number of documents in input file
	private int uniqueWordCount;	//Total number of words in input file
	private ArrayList<String> uniqueWordList;	//Stores the unique words across all documents in input data
	private ArrayList<String> documentClassList;	//Stores class of each document
	private ArrayList<Integer> n_k;	//Used to calculate tf x idf
	private VectorEntry[][] dataVector;	//Stores term frequency and tf x idf values in an enclosing class
	private ArrayList<String> uniqueClassList;  //The unique classes present in the input file.
	
	//Output tables...
	private double[][] tf_ed_table;
	private double[][] tf_cos_table;
	private double[][] tf_idf_ed_table;
	private double[][] tf_idf_cos_table;

    //Constructor
    public Vector(String inputFile) {
		createWordAndClassLists(inputFile);
		this.dataVector = new VectorEntry[documentCount][uniqueWordCount];
		for (int i = 0; i<this.dataVector.length; i++) {
			for (int j = 0; j<this.dataVector[i].length; j++) {
				this.dataVector[i][j] = new VectorEntry();
			}
		}
		buildTFVector(inputFile);
		buildTF_IDFVector();
	}
    
	//Determine the number of documents and the number of unique words in the file.
    //Then build uniqueWord, uniqueClass, and documentClass lists.
	private void createWordAndClassLists(String inputFile) {
		System.out.println("Calculating Unique Word Count and Document Count...\n---------------------------------------------------");
		BufferedReader reader = null;
		int documentCount = 0;
		ArrayList<String> uniqueWordList = new ArrayList<String>();
		ArrayList<String> classNamesList = new ArrayList<String>();
		try {
			File file = new File(inputFile);
			reader = new BufferedReader(new FileReader(file));
			String line;
			while ((line = reader.readLine()) != null) {
				documentCount++;
				String[] classSplit = line.split("\t");
				String[] wordsInDocument = classSplit[1].split(" ");
				for (int i = 0; i<wordsInDocument.length; i++) {
					if (!uniqueWordList.contains(wordsInDocument[i])) {
						uniqueWordList.add(wordsInDocument[i]);
					}
				}
				classNamesList.add(classSplit[0]);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		this.documentCount = documentCount;
		this.uniqueWordCount = uniqueWordList.size();
		Collections.sort(uniqueWordList);	//Searching this will be faster if I use binary search. Gotta sort it to use that.
		this.uniqueWordList = uniqueWordList;
		this.documentClassList = classNamesList;
		this.uniqueClassList = uniqueClassListGeneration(this.documentClassList);
		System.out.println("Unique Word Count from List: "+uniqueWordList.size());
		System.out.println("Document Count: "+ documentCount);
		System.out.println();
	}
    
	//Create the list of unique classes
    //Our full dataset has 8 or so unique classes
	private ArrayList<String> uniqueClassListGeneration(ArrayList<String> classList) {
		ArrayList<String> uniqueClassList = new ArrayList<String>();
		for (int i = 0; i<classList.size(); i++) {
			if (!uniqueClassList.contains(classList.get(i))) {
				uniqueClassList.add(classList.get(i));
			}
		}
		Collections.sort(uniqueClassList);
		return uniqueClassList;
	}
	
    //Builds the term frequency matrices/ data structures, or whatever you wanna call em
	private void buildTFVector(String inputFile) {
		System.out.println("Building Term Frequency Vector...\n---------------------------------");
		BufferedReader reader = null;
		try {
			File file = new File(inputFile);
			reader = new BufferedReader(new FileReader(file));
			String line;
			int documentPointer = 0;
			while ((line = reader.readLine()) != null) {
				String[] classSplit = line.split("\t");
				String[] wordsInDocument = classSplit[1].split(" ");
				for (int i = 0; i<wordsInDocument.length; i++) {
					int index = Collections.binarySearch(this.uniqueWordList, wordsInDocument[i]);
					this.dataVector[documentPointer][index].incrementTermFrequency();
				}
				documentPointer++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		n_k = new ArrayList<Integer>();
		for (int i = 0; i<this.uniqueWordCount; i++) {
			int count = 0;
			for (int j = 0; j<this.documentCount; j++) {
				if (this.dataVector[j][i].gettf() != 0) {
					count++;
				}
			}
			this.n_k.add(count);
		}
		System.out.println("Done\n");
	}
	
    //Builds inverse term frequency vector/ matrix
	private void buildTF_IDFVector() {
		System.out.println("Building Inverse Term Frequency Vector...\n-----------------------------------------");
		for (int i = 0; i<this.dataVector.length; i++) {
			for (int j = 0; j<this.dataVector[i].length; j++) {
				this.dataVector[i][j].tf_idf_calculation(this.n_k.get(j), this.documentCount);
			}
		}
		System.out.println("Done\n");
	}
	
    //Euclidean measure between two documents stored in a class called SimilarityMeasure
	private SimilarityEnclosure euclideanMeasure(VectorEntry[][] dataVector, int document1, int document2, boolean tf_idf) {
		double similarity = 0;
		if (tf_idf) {
			for (int j = 0; j<this.uniqueWordCount; j++) {
				similarity += Math.pow(this.dataVector[document1][j].get_tf_idf()-this.dataVector[document2][j].get_tf_idf(), 2);
			}
		} else {
			for (int j = 0; j<this.uniqueWordCount; j++) {
				similarity += Math.pow(this.dataVector[document1][j].gettf()-this.dataVector[document2][j].gettf(), 2);
			}
		}
		similarity = Math.sqrt(similarity);
		SimilarityEnclosure sim = new SimilarityEnclosure(document1, document2, similarity);
		return sim;
	}
	
    //Cosine similarity measure between two documents stored in a class called SimilarityMeasure
	private SimilarityEnclosure cosineMeasure(VectorEntry[][] dataVector, int document1, int document2, boolean tf_idf) {
        int dot_product = 0;
        double d1_norm = 0;
        double d2_norm = 0;
        for (int j = 0; j<this.uniqueWordCount; j++) {
            if (tf_idf) {
                dot_product += this.dataVector[document1][j].get_tf_idf()*this.dataVector[document2][j].get_tf_idf();
                d1_norm += Math.pow(this.dataVector[document1][j].get_tf_idf(), 2);
                d2_norm += Math.pow(this.dataVector[document2][j].get_tf_idf(), 2);
            } else {
                dot_product += this.dataVector[document1][j].gettf()*this.dataVector[document2][j].gettf();
                d1_norm += Math.pow(this.dataVector[document1][j].gettf(), 2);
                d2_norm += Math.pow(this.dataVector[document2][j].gettf(), 2);
            }
        }
        
        //Build enclosing data structure
        double cosine = dot_product/(d1_norm*d2_norm);
        SimilarityEnclosure sim = new SimilarityEnclosure(document1, document2, cosine);
        return sim;
	}
	
    //This class does the take-one-out cross validation tests for the classifier to determine its accuracy
    //This specific one is for the Euclidean distance similarity measure
	private String euclid_cross_validation(boolean tf_idf) {		
		double[][] outputTable = new double[this.uniqueClassList.size()+1][3];
		if (tf_idf) {
			System.out.println("Running Inverse Term Frequency-Euclidean Distance Cross Validation Tests...\n-------------------------------------------------------------------");
		} else {
			System.out.println("Running Term Frequency-Euclidean Distance Cross Validation Tests...\n-------------------------------------------------------------------");
		}
		double overallAccuracy = 0;
		ArrayList<SimilarityEnclosure> similarityMeasure = new ArrayList<SimilarityEnclosure>(this.uniqueWordCount); 
		int documentAccuracy=0;
		for (int i = 0; i<this.documentCount; i++) {
			documentAccuracy = 0;
			for (int j = 0; j<this.documentCount; j++) {
				if (j != i) {
					if (tf_idf) {
						similarityMeasure.add(euclideanMeasure(this.dataVector, i, j, true));
					} else {
						similarityMeasure.add(euclideanMeasure(this.dataVector, i, j, false));
					}
				}
			}
			Collections.sort(similarityMeasure);
			SimilarityEnclosure smallest = similarityMeasure.get(0);
			String trueClass = this.documentClassList.get(i);
			String predictedClass = this.documentClassList.get(smallest.document2);
			if (trueClass.equals(predictedClass)) {
				int index = Collections.binarySearch(this.uniqueClassList, trueClass);
				outputTable[index][0]++;
				documentAccuracy++;
			} else {
				int index = Collections.binarySearch(this.uniqueClassList, trueClass);
				outputTable[index][1]++;
			}
			for (int k = 0; k<this.uniqueClassList.size(); k++) {
				String word = this.uniqueClassList.get(k);
				int counter = 0;
				for (int l = 0; l<this.documentClassList.size(); l++) {
					if (this.documentClassList.get(l).equals(word)) {
						counter++;
					}
				}
				outputTable[k][2] = outputTable[k][0]/counter;
			}
			overallAccuracy+=documentAccuracy;
			similarityMeasure.clear();
		}
		for (int i = 0; i<3; i++) {
			double counter = 0;
			for (int j = 0; j<this.uniqueClassList.size(); j++) {
				counter+=outputTable[j][i];
			}
			outputTable[this.uniqueClassList.size()][i] = counter;
		}
		outputTable[this.uniqueClassList.size()][2] = outputTable[this.uniqueClassList.size()][0]/(outputTable[this.uniqueClassList.size()][1]+outputTable[this.uniqueClassList.size()][0]-1);
		if (tf_idf) {
			this.tf_idf_ed_table = outputTable;
		} else {
			this.tf_ed_table = outputTable;
		}
		overallAccuracy /= this.documentCount-1;
		StringBuilder sb = new StringBuilder();
		sb.append((overallAccuracy*100)+"%");
		return sb.toString();
	}
	
    //This class does the take-one-out cross validation tests for the classifier to determine its accuracy
    //This specific one is for the Cosine similarity measure
	private String cosine_cross_validation(boolean tf_idf) {
		double[][] outputTable = new double[this.uniqueClassList.size()+1][3];  //Store output information in this
		if (tf_idf) {
			System.out.println("Running Inverse Term Frequency-Cosine Distance Cross Validation Tests...\n-------------------------------------------------------------------");
		} else {
			System.out.println("Running Term Frequency-Cosine Distance Cross Validation Tests...\n-------------------------------------------------------------------");
		}
		double overallAccuracy = 0;
		ArrayList<SimilarityEnclosure> similarityMeasure = new ArrayList<SimilarityEnclosure>(this.uniqueWordCount);    //Store list of similarityMeasures between different documents in this ArrayList
		int documentAccuracy=0; //Keep track of current document i's accuracy value
		for (int i = 0; i<this.documentCount; i++) {
			documentAccuracy = 0;
			for (int j = 0; j<this.documentCount; j++) {
				if (j != i) {
                    //Calculate cosine similarity between documents i and j
					if (tf_idf) {
						similarityMeasure.add(cosineMeasure(this.dataVector, i, j, true));
					} else {
						similarityMeasure.add(cosineMeasure(this.dataVector, i, j, false));
					}
				}
			}
            //Sort ArrayList and find largest similarity measure for cosine
			Collections.sort(similarityMeasure, Collections.reverseOrder());
			SimilarityEnclosure smallest = similarityMeasure.get(0);
			String trueClass = this.documentClassList.get(i);
			String predictedClass = this.documentClassList.get(smallest.document2);
			//Determine if prediction was accurate or not
            if (trueClass.equals(predictedClass)) {
				int index = Collections.binarySearch(this.uniqueClassList, trueClass);
				outputTable[index][0]++;
				documentAccuracy++;
			} else {
				int index = Collections.binarySearch(this.uniqueClassList, trueClass);
				outputTable[index][1]++;
			}
			for (int k = 0; k<this.uniqueClassList.size(); k++) {
				String word = this.uniqueClassList.get(k);
				int counter = 0;
				for (int l = 0; l<this.documentClassList.size(); l++) {
					if (this.documentClassList.get(l).equals(word)) {
						counter++;
					}
				}
				outputTable[k][2] = outputTable[k][0]/counter;
			}
			overallAccuracy+=documentAccuracy;
			similarityMeasure.clear();
		}
		for (int i = 0; i<3; i++) {
			double counter = 0;
			for (int j = 0; j<this.uniqueClassList.size(); j++) {
				counter+=outputTable[j][i];
			}
			outputTable[this.uniqueClassList.size()][i] = counter; 
		}
		outputTable[this.uniqueClassList.size()][2] = outputTable[this.uniqueClassList.size()][0]/(outputTable[this.uniqueClassList.size()][1]+outputTable[this.uniqueClassList.size()][0]-1); 
		if (tf_idf) {
			this.tf_idf_cos_table = outputTable;
		} else {
			this.tf_cos_table = outputTable;
		}
		overallAccuracy /= this.documentCount-1;
		StringBuilder sb = new StringBuilder();
		sb.append((overallAccuracy*100)+"%");
		return sb.toString();
	}
    
	//Creates output tables
	public String outputTable(boolean ed, boolean tf_idf) {
		StringBuilder sb = new StringBuilder();
		if (tf_idf) {
			if (ed) {
				sb.append("TF*IDF+ED:").append(System.lineSeparator());
				sb.append("Class Name|# Correct|# Incorrect|Accuracy|").append(System.lineSeparator());
				for (int i = 0; i<this.uniqueClassList.size()+1; i++) {
					if (i <uniqueClassList.size()) {
						sb.append(String.format("%-10s: ",uniqueClassList.get(i) ));
					} else {
						sb.append(String.format("%-10s: ","OVERALL"));
					}
					for (int j = 0; j<3; j++) {
						sb.append(String.format("%7.2f |", this.tf_idf_ed_table[i][j]));
					}
					sb.append(System.lineSeparator());
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			} else {
				sb.append("TF*IDF+COS:").append(System.lineSeparator());
				sb.append("Class Name|# Correct|# Incorrect|Accuracy|").append(System.lineSeparator());
				for (int i = 0; i<this.uniqueClassList.size()+1; i++) {
					if (i <uniqueClassList.size()) {
						sb.append(String.format("%-10s: ",uniqueClassList.get(i) ));
					} else {
						sb.append(String.format("%-10s: ","OVERALL"));
					}
					for (int j = 0; j<3; j++) {
						sb.append(String.format("%7.2f |", this.tf_idf_cos_table[i][j]));
					}
					sb.append(System.lineSeparator());
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			}
		} else {
			if (ed) {
				sb.append("TF+ED:").append(System.lineSeparator());
				sb.append("Class Name|# Correct|# Incorrect|Accuracy|").append(System.lineSeparator());
				for (int i = 0; i<this.uniqueClassList.size()+1; i++) {
					if (i <uniqueClassList.size()) {
						sb.append(String.format("%-10s: ",uniqueClassList.get(i) ));
					} else {
						sb.append(String.format("%-10s: ","OVERALL"));
					}
					for (int j = 0; j<3; j++) {
						sb.append(String.format("%7.2f |", this.tf_ed_table[i][j]));
					}
					sb.append(System.lineSeparator());
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			} else {
				sb.append("TF+COS:").append(System.lineSeparator());
				sb.append("Class Name|# Correct|# Incorrect|Accuracy|").append(System.lineSeparator());
				for (int i = 0; i<this.uniqueClassList.size()+1; i++) {
					if (i <uniqueClassList.size()) {
						sb.append(String.format("%-10s: ",uniqueClassList.get(i) ));
					} else {
						sb.append(String.format("%-10s: ","OVERALL"));
					}
					for (int j = 0; j<3; j++) {
						sb.append(String.format("%7.2f |", this.tf_cos_table[i][j]));
					}
					sb.append(System.lineSeparator());
				}
				sb.append(System.lineSeparator()).append(System.lineSeparator());
			}
		}
		return sb.toString();
	}
	
    //Private nested class that
	private static class SimilarityEnclosure implements Comparable<SimilarityEnclosure>{
		int document1;
		int document2;
		double similarity;
		
		public SimilarityEnclosure(int document1, int document2, double similarityMeasure) {
			this.document1 = document1;
			this.document2 = document2;
			this.similarity = similarityMeasure;
		}

		@Override
		public int compareTo(SimilarityEnclosure o) {
			if (this.similarity < o.similarity) {
				return -1;
			} else if (this.similarity == o.similarity) {
				return 0;
			} else {
				return 1;
			}
		}
	}
	
    //Run the main clause
	public static void main(String[] args) {
		long start = System.currentTimeMillis();    //Keep track of time
		Vector vector = new Vector(args[0]);    //Input from command line
		String euclid_tf = vector.euclid_cross_validation(false);
		String euclid_tf_idf = vector.euclid_cross_validation(true);
		String cosine_tf = vector.cosine_cross_validation(false);
		String cosine_tf_idf = vector.cosine_cross_validation(true);
		System.out.println();
		System.out.println("Results:\n---------\nOverview:\n---------");
		System.out.println("Method:   Accuracy");
		System.out.println("TF+ED: "+ euclid_tf);
		System.out.println("TF+COS: "+ cosine_tf);
		System.out.println("TF*IDF+ED: "+ euclid_tf_idf);
		System.out.println("TF*IDF+COS: "+ cosine_tf_idf);
		System.out.println("\n");
		System.out.println("Case by Case Tables:\n--------------------");
		System.out.println(vector.outputTable(true, false));
		System.out.println(vector.outputTable(false, false));
		System.out.println(vector.outputTable(true, true));
		System.out.println(vector.outputTable(false, true));
		long end = System.currentTimeMillis();
		long elapsed = end-start;
		System.out.println("Elapsed Seconds: "+elapsed/1000);   //Calculate elapsed time
	}

}
