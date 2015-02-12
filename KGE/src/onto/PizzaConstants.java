package onto;

public class PizzaConstants {
	/**
	 * Name Spaces
	 */
	public static final String PIZZA_NS = "http://www.co-ode.org/ontologies/pizza/pizza.owl#";
	
	/**
	 * Class Names
	 */
	public static final String PIZZA_PIZZA = "Pizza"; 
	public static final String PIZZA_PIZZABASE = "PizzaBase"; 
	public static final String PIZZA_PIZZATOPPING = "PizzaTopping"; 
	public static final String PIZZA_COUNTRY = "Country"; 
	public static final String PIZZA_SPICINESS = "Spiciness"; 
	
	public static final String PIZZA_PIZZA_CLS = getWithNS(PIZZA_PIZZA);
	public static final String PIZZA_PIZZABASE_CLS = getWithNS(PIZZA_PIZZABASE);
	public static final String PIZZA_PIZZATOPPING_CLS = getWithNS(PIZZA_PIZZATOPPING);
	public static final String PIZZA_COUNTRY_CLS = getWithNS(PIZZA_COUNTRY);
	public static final String PIZZA_SPICINESS_CLS = getWithNS(PIZZA_SPICINESS);
			
	public static String getWithNS(String name) {
		return PIZZA_NS + name;
	}
	
	public static String getLocalName(String iriStr) {
		return iriStr.replace("http://www.co-ode.org/ontologies/pizza/pizza.owl#", "");
	}
}
