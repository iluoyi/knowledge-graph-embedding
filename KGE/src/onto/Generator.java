package onto;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLIndividual;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;

import utils.IO;

import com.clarkparsia.pellet.owlapiv3.PelletReasoner;
import com.clarkparsia.pellet.owlapiv3.PelletReasonerFactory;

public class Generator {
	public static final int NUM_OF_PIZZA = 2000; // for each type of pizza
	public static final int NUM_OF_PIZZABASE = NUM_OF_PIZZA * 12; // for each type of base
	public static final int NUM_OF_PIZZATOPPING = NUM_OF_PIZZABASE; // for each type of topping
	public static final int MAX_NUM_OF_HASPIZZATOPPING = 5;
	
	public static final int NUM_OF_COUNTRAY = 200; // total number of countries
	public static final int NUM_OF_SPICINESS = 200; // total number of spiciness
	
	public static final String TRAIN_FILE = "datasets/Pizza8/train.txt";
	public static final String VALID_TEST_FILE = "datasets/Pizza8/valid_test.txt";
	
	OWLOntology ontology = null;
	OWLDataFactory df = null;
	PelletReasoner reasoner = null;
	
	ArrayList<OWLIndividual> pizzaList = new ArrayList<OWLIndividual>();
	ArrayList<OWLIndividual> pizzaBaseList = new ArrayList<OWLIndividual>();
	ArrayList<OWLIndividual> pizzaToppingList = new ArrayList<OWLIndividual>();
	ArrayList<OWLIndividual> countryList = new ArrayList<OWLIndividual>();
	ArrayList<OWLIndividual> spicyList = new ArrayList<OWLIndividual>();
	
	public boolean loadOntology(String localName) {
		try {
			File file = new File(localName);
			IRI iri = IRI.create(file.toURI().toURL());
			OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
			ontology = manager.loadOntology(iri);
			if (ontology != null) {
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		} 
		return false;
	}
	
	public void makeIndividuals(OWLClass rootCls, HashSet<OWLClass> visitedCls, ArrayList<OWLIndividual> individualSet, int numOfIns) {
		Set<OWLClass> subClas = reasoner.getSubClasses(rootCls, true).getFlattened();
		Iterator<OWLClass> it = subClas.iterator();
		OWLClass firstElement = it.next();
		if (firstElement.toString().equals("owl:Nothing")) { // leaf node
			for (int i = 0; i < numOfIns; i ++) {
				String individualName = PizzaConstants.getLocalName(rootCls.asOWLClass().getIRI().toString()) + "_" + i;
				OWLIndividual newIns = df.getOWLNamedIndividual(IRI.create(individualName));
				individualSet.add(newIns);
			}
		} else {
			visitedCls.add(rootCls);
			for (OWLClass eachCls : subClas) {
				if (!visitedCls.contains(eachCls)) {
					visitedCls.add(eachCls);
					makeIndividuals(eachCls, visitedCls, individualSet, numOfIns);
				}
			}
		}
	}
	
	public void makePizza(OWLClass pizzaCls, ArrayList<OWLIndividual> pizzaList) {
		makeIndividuals(pizzaCls, new HashSet<OWLClass>(), pizzaList, NUM_OF_PIZZA);
	}
	
	public void makePizzaBase(OWLClass pizzaBaseCls, ArrayList<OWLIndividual> pizzaBaseList) {
		makeIndividuals(pizzaBaseCls, new HashSet<OWLClass>(), pizzaBaseList, NUM_OF_PIZZABASE);
	}
	
	public void makePizzaTopping(OWLClass pizzaToppingCls, ArrayList<OWLIndividual> pizzaToppingList) {
		makeIndividuals(pizzaToppingCls, new HashSet<OWLClass>(), pizzaToppingList, NUM_OF_PIZZATOPPING);
	}
	
	public void makeCountry(OWLClass countryCls, ArrayList<OWLIndividual> countryList) {
		makeIndividuals(countryCls, new HashSet<OWLClass>(), countryList, NUM_OF_COUNTRAY);
	}
	
	public void makeSpiciness(OWLClass spicyCls, ArrayList<OWLIndividual> spicyList) {
		makeIndividuals(spicyCls, new HashSet<OWLClass>(), spicyList, NUM_OF_SPICINESS);
	}
	
	public void generateIndividuals() {
		if (ontology == null) {
			System.out.println("Loading Ontology failed...");
		} else {
			df = ontology.getOWLOntologyManager().getOWLDataFactory();
			reasoner = PelletReasonerFactory.getInstance().createReasoner(ontology);
			
			// make Pizza individuals
			System.out.println("Making pizzas...");
			OWLClass pizzaCls = df.getOWLClass(IRI.create(PizzaConstants.PIZZA_PIZZA_CLS));
			makePizza(pizzaCls, pizzaList);
			
			// make PizzaBase individuals
			System.out.println("Making pizza bases...");
			OWLClass pizzaBaseCls = df.getOWLClass(IRI.create(PizzaConstants.PIZZA_PIZZABASE_CLS));
			makePizzaBase(pizzaBaseCls, pizzaBaseList);
			
			// make PizzaTopping individuals
			System.out.println("Making pizza toppings...");
			OWLClass pizzaToppingCls = df.getOWLClass(IRI.create(PizzaConstants.PIZZA_PIZZATOPPING_CLS));
			makePizzaTopping(pizzaToppingCls, pizzaToppingList);
		
			// make Country individuals
			System.out.println("Making contries...");
			OWLClass countryCls = df.getOWLClass(IRI.create(PizzaConstants.PIZZA_COUNTRY_CLS));
			makeCountry(countryCls, countryList);
			
			// make Spiciness individuals
			System.out.println("Making spiciness...");
			OWLClass spicyCls = df.getOWLClass(IRI.create(PizzaConstants.PIZZA_SPICINESS_CLS));
			makeCountry(spicyCls, spicyList);
		}
	}
	
	public void generateTriples() {
		/*
		 * Rules:
		 * 1. Each pizza has 0 or 1 origin country; 
		 * 2. Each pizza has only 1 PizzaBase;
		 * 3. Each pizza has (0, MAX_NUM_OF_HASPIZZATOPPING] PizzaToppings;
		 * 4. Each pizza topping has 0 or 1 spicy degree.
		 */
		Random rand = new Random();
		
		ArrayList<OWLIndividual> remainingPizzaBase = new ArrayList<OWLIndividual>(pizzaBaseList);
		ArrayList<OWLIndividual> remainingPizzaTopping = new ArrayList<OWLIndividual>(pizzaToppingList);
		
		IO ioTrain = null, ioTest = null;
		int index, numOfTopping;
		
		try {
			ioTrain = new IO(TRAIN_FILE, "w");		
			ioTest = new IO(VALID_TEST_FILE, "w");
			
			System.out.println("Generating triples pizzas...");
			for (OWLIndividual pizza : pizzaList) {
				//System.out.println("current: " + pizza);
				if (rand.nextBoolean()) {
					index = rand.nextInt(countryList.size());
					IO.assignTrainOrTest(ioTrain, ioTest, pizza + "\thasCountryOrigin\t" + countryList.get(index), rand.nextFloat(), 0.8F);
				}

				if (remainingPizzaBase.size() <= 0) {
					System.out.println("Lack of pizza base!");
					System.exit(1);
				} else {
					index = rand.nextInt(remainingPizzaBase.size());
					IO.assignTrainOrTest(ioTrain, ioTest, pizza + "\thasBase\t" + remainingPizzaBase.get(index), rand.nextFloat(), 0.8F);
					IO.assignTrainOrTest(ioTrain, ioTest, pizza + "\thasIngredient\t" + remainingPizzaBase.get(index), rand.nextFloat(), 0.8F);
					IO.assignTrainOrTest(ioTrain, ioTest, remainingPizzaBase.get(index) + "\tisBaseOf\t" + pizza, rand.nextFloat(), 0.8F);
					IO.assignTrainOrTest(ioTrain, ioTest, remainingPizzaBase.get(index) + "\tisIngredientOf\t" + pizza, rand.nextFloat(), 0.8F);
					remainingPizzaBase.remove(index);
				}
				
				if (remainingPizzaTopping.size() <= 0) {
					System.out.println("Lack of pizza topping!");
					System.exit(1);
				} else {
					numOfTopping  = rand.nextInt(MAX_NUM_OF_HASPIZZATOPPING + 1);
					for (int i = 0; i < numOfTopping; i ++) {
						index = rand.nextInt(remainingPizzaTopping.size());
						IO.assignTrainOrTest(ioTrain, ioTest, pizza + "\thasTopping\t" + remainingPizzaTopping.get(index), rand.nextFloat(), 0.9F);
						IO.assignTrainOrTest(ioTrain, ioTest, pizza + "\thasIngredient\t" + remainingPizzaTopping.get(index), rand.nextFloat(), 0.9F);
						IO.assignTrainOrTest(ioTrain, ioTest, remainingPizzaTopping.get(index) + "\tisToppingOf\t" + pizza, rand.nextFloat(), 0.9F);
						IO.assignTrainOrTest(ioTrain, ioTest, remainingPizzaTopping.get(index) + "\tisIngredientOf\t" + pizza, rand.nextFloat(), 0.9F);
						remainingPizzaTopping.remove(index);
					}
				}
			}
			
			for (OWLIndividual topping : pizzaToppingList) {
				index = rand.nextInt(spicyList.size());
				IO.assignTrainOrTest(ioTrain, ioTest, topping + "\thasSpiciness\t" + spicyList.get(index), rand.nextFloat(), 0.9F);
			}
			
			ioTrain.writeClose();
			ioTest.writeClose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public int getNumOfEntity() {
		return pizzaList.size() + pizzaBaseList.size() + pizzaToppingList.size() + countryList.size() + spicyList.size();
	}
	
	public static void main(String args[]) {
		Generator gen = new Generator();
		gen.loadOntology("ontology/pizza.owl");
		gen.generateIndividuals();
		gen.generateTriples();
		System.out.println("Entity size: " + gen.getNumOfEntity());
	}
}
