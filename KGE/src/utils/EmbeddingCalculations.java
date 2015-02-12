package utils;

public class EmbeddingCalculations {
	public static float norm(float[] embedding, String norm) {
		float mode = 0.0f;
		int dimension = embedding.length;
		
		if (norm.equals("L1")) {
			for (int i = 0; i < dimension; i++) {
				mode += Math.abs(embedding[i]);
			}
		} else if (norm.equals("L2")) {
			for (int i = 0; i < dimension; i++) {
				mode += embedding[i] * embedding[i];
			}
			// mode = (float)Math.sqrt(mode);
		} else {
		}
		return mode;
	}

	public static float[] embCalculator(float[] firstEmb, String operator, float[] secondEmb) {
		int dimension = firstEmb.length;
		float[] resultEmb = new float[dimension];
		if (operator.equals("+")) {
			for (int i = 0; i < dimension; i++) {
				resultEmb[i] = firstEmb[i] + secondEmb[i];
			}
		}
		else if (operator.equals("-")) {
			for (int i = 0; i < dimension; i++) {
				resultEmb[i] = firstEmb[i] - secondEmb[i];
			}
		} else {

		}
		return resultEmb;
	}

	public static float[] embCalculator(float[] firstEmb, String operator, float second) {
		int dimension = firstEmb.length;
		float[] resultEmb = new float[dimension];
		if (operator.equals("*")) {
			for (int i = 0; i < dimension; i++) {
				resultEmb[i] = firstEmb[i] * second;
			}
		}
		else if (operator.equals("/")) {
			for (int i = 0; i < dimension; i++) {
				resultEmb[i] = firstEmb[i] / second;
			}
		} else {

		}
		return resultEmb;
	}

	public static float[] getGradientEmb(float[] distanceEmb, String norm) {
		int dimension = distanceEmb.length;
		float[] gradientEmb = new float[dimension];
		if (norm.equals("L1")) {
			for (int i = 0; i < dimension; i++) {
				if (distanceEmb[i] > 0.0f)
					gradientEmb[i] = 1.0f;
				else
					gradientEmb[i] = -1.0f;
			}
		} else if (norm.equals("L2")) {
			for (int i = 0; i < dimension; i++) {
				gradientEmb[i] = 2.0f * distanceEmb[i];
			}
		}
		return gradientEmb;
	}

	public static float[] getDistanceEmb(float[] headEmb, float[] relationEmb, float[] tailEmb) {
		return embCalculator(embCalculator(headEmb, "+", relationEmb), "-", tailEmb);
	}
	
	public static void displayEmb(float[] embedding) {
		for (int i = 0; i < embedding.length; i++) {
			System.out.print(embedding[i] + ",");
		}
		System.out.println();
	}
}
