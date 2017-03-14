# A simple graph

Create and connect a simple graph using the following code

```java
Graph g = new GraphBuilder().build();
g.connect(isconnected-> {
	//your next code goes here...
});
```

Next, create a first node and set some attributes:

```java
Node sensor0 = g.newNode(0, 0); // for now, use 0,0 as params
sensor0.set("id", "4494F");
sensor0.set("name", "sensor0"); 
```

Then, create a second node and connect the two nodes with a relation

```java
Node room0 = g.newNode(0, 0); // for now, use 0,0 as params
room0.set("name", "room0");
room0.add("sensors", sensor0);
```

Finally, relations can be traversed as follows

```java
room0.rel("sensors", (Node[] sensors) -> { //iterate over saved sensors
	System.out.println("Sensors");
	for (int i = 0; i < sensors.length; i++) {
		System.out.println("\t" + sensors[i].toString());
	}
});
```

Note that graph processing and traversing is fully asynchronous.
