# Persistence 

MWG comes with a built-in persistence mechanism. Nodes and relationships are automatically serialized and unserialized by MWG and can be stored in different key/value stores. The following example shows how MWG is configured to use Google's LevelDB as a storage backend.  

```java
Graph g = new GraphBuilder()
	.withMemorySize(10000) //cache size before sync to disk
	.withStorage(new LevelDBStorage("mwg_db"))
	.build();
       
g.connect(isConnected -> {
	Node sensor = g.newNode(0, System.currentTimeMillis()); //create a new node for world 0 and time 0
	sensor.set("id", Math.abs(rand.nextInt()));
	sensor.set("value", rand.nextInt());
	g.index("sensors", sensor, "id", res -> {
		g.save(saveResult -> {
			g.findAll(0, System.currentTimeMillis(), "sensors", allSensorsNow -> {
				System.out.println("All sensors indexed:");
				for (Node sensorNow : allSensorsNow) {
					System.out.println("\t" + sensorNow.toString());
				}
			});
		});
	});
});
```