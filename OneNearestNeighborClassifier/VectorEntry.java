/**
 * This class holds information about the entry that is being placed into the 2D array
 * that holds document word frequency information.
 * @author PavanVittala
 */

public class VectorEntry {
	private int tf;		//Term frequency
	private double tf_idf;	//Inverse term frequency
	
	public VectorEntry() {
		this.tf = 0;
		this.tf_idf = 0;
	}
	
	public int gettf() {
		return tf;
	}
	
	public double get_tf_idf() {
		return this.tf_idf;
	}
	
	/**
	 * Increment tf every time a new instance of the word is seen in the document
	 */
	public void incrementTermFrequency() {
		this.tf++;
	}
	
	/**
	 * Function that calculates tf x idf for a given V(i,j) index
	 * @param n_k: number of documents in the collection that contain term_k
	 * @param totalDocuments: Total number of documents in the input file
	 */
	public void tf_idf_calculation(int n_k, int totalDocuments) {
		double idf = Math.log10(totalDocuments/n_k);
		this.tf_idf = tf*idf;
	}
}
