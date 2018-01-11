
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer

object App {

  case class Violation(time_of_stop: String, agency: String, description: String, location: String,accident: String, belts: String, personal_injury: String, property_damage: String, fatal: String, commercial_license: String, hazmat: String, commercial_vehicle: String, alcohol: String,
                       work_zone: String, state: String, vehicle_type: String, year: String, make: String, model: String, color: String, violation_type: String, contributed_to_accident: String, race: String, gender: String, driver_city: String, driver_state: String, dl_state: String)

  // EXAMPLE INSTANCE:
  // 09/24/2013,17:11:00,MCP,"3rd district, Silver Spring",DRIVING VEHICLE ON HIGHWAY WITH SUSPENDED REGISTRATION,8804 FLOWER AVE,,,No,No,No,No,No,No,No,No,No,No,MD,
  // 02 - Automobile,2008,FORD,4S,BLACK,Citation,13-401(h),Transportation Article,No,BLACK,M,TAKOMA PARK,MD,MD,A - Marked Patrol,

  def main(args: Array[String]): Unit = {

    val PATH = "data/Traffic_Violations.csv"

    val session = SparkSession.builder.master("local").appName("CS450-Project").config("spark.some.config.option","config-value").getOrCreate()
    val sc = session.sparkContext
    val sQLContext = new SQLContext(sc)

    val data = sc.textFile(PATH)
    val header = data.first()
    val rows = data.filter(l => l != header).map(line => line.substring(0,line.length-4))
    val allSplit = rows.map(line => line.split(";"))

    val allSplit2 = allSplit.filter(arr => (arr.length == 35))map(arr => remove(arr, arr.length - 1)) // remove geo-location
    val allSplit3 = allSplit.filter(arr => (arr.length == 34)) // instances that does not have geo-location

    val finalSplits = allSplit2.union(allSplit3)


    val allData = finalSplits.map(p => Violation(p(1).toString,p(2).toString,p(4).toString,p(5).toString,p(8).toString,p(9).toString,p(10).toString,
      p(11).toString,p(12).toString,p(13).toString,p(14).toString,p(15).toString,p(16).toString,p(17).toString,p(18).toString,p(19).toString,p(20).toString,p(21).toString,p(22).toString,p(23).toString,
      p(24).toString,p(27).toString,p(28).toString,p(29).toString,p(30).toString,p(31).toString,p(32).toString))


     // allData.foreach(v => print(v.year + " " + v.make + " " + v.model + "\n"))


    // PRE-PROCESSING

    val daytimeRDD = allData.map(v => stopTime(v.time_of_stop)) // this RDD stores day-times by considering the stop times.
    val vehicleTypeRDD = allData.map(v => vehicleType(v.vehicle_type))
    val statesRDD = allData.map(v => v.state).filter(st => !st.equals(" ")).filter(st => (st.length() < 3 && st.length() > 1)).map(st => stateStatus(st))
    val yearRDD = allData.map(v => validYear(v.year))
    //printUniqueElements_int(yearRDD)
    // Adamlar rastgele yıl sallamışlar bazı modeller için. Bunu kullanmamak en doğrusu. 3503 falan var
    val raceRDD = allData.map(v => v.race).map(racex => raceCategory(racex))
    //raceRDD.foreach(y => print(y.toString + " \n"))
    val genderRDD = allData.map(v => v.gender).map(gender => genderCategory(gender))
    //genderRDD.foreach(y => print(y.toString + " \n"))
    val violationTypeRDD = allData.map(v => v.violation_type).map(vtype => violationCategory(vtype))
    //violationTypeRDD.foreach(y => print(y.toString + " \n"))
    // between features 8-17 (YES/NO)
    val data_8_17_v1 = allData.map(v => (v.accident, v.belts, v.personal_injury, v.property_damage, v.fatal, v.commercial_license, v.hazmat, v.commercial_vehicle, v.alcohol, v.work_zone))
    val data_8_17RDD = data_8_17_v1.map(v => (boolCategory(v._1),boolCategory(v._2),boolCategory(v._3),boolCategory(v._4),boolCategory(v._5),boolCategory(v._6),boolCategory(v._7),boolCategory(v._8),boolCategory(v._9),boolCategory(v._10)))
    //data_8_17RDD.foreach(y => print(y.toString + " \n"))
    val contributedAccidentRDD = allData.map(v => boolCategory(v.contributed_to_accident))
   // contributedAccidentRDD.foreach(y => print(y.toString + " \n"))




    // Indexing text-based features of the data

    import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
    import sQLContext.implicits._


    //-------------- COLOR ----------------------//

    val color_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("color_index")
    val colorRDD = allData.map(v => v.color)
    val indexedColorDF = color_indexer.fit(colorRDD.toDF()).transform(colorRDD.toDF())
    //indexedColorDF.show()
    val color_converter = new IndexToString()
      .setInputCol("color_index")
      .setOutputCol("originalCategory")
    val converted_color = color_converter.transform(indexedColorDF)
    //converted_color.select("value", "color_index", "originalCategory").show()
    val indexedColorRDD   = indexedColorDF.select("color_index").rdd
    //indexedColorRDD.foreach(y => print(y.toString + " \n"))
    val final_color_RDD = indexedColorRDD.map(row => row(0))
    //final_color_RDD.foreach(y => print(y.toString + " \n"))


    //-------------- MAKE  ------------------//

    val make_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("make_index")
    val makeRDD = allData.map(v => v.make)
    val indexedMakeDF = make_indexer.fit(makeRDD.toDF()).transform(makeRDD.toDF())
    //indexedMakeDF.show()
    val make_converter = new IndexToString()
      .setInputCol("make_index")
      .setOutputCol("originalCategory")
    val converted_make = make_converter.transform(indexedMakeDF)
    //converted_make.select("value", "make_index", "originalCategory").show()
    val indexedMakeRDD   = indexedMakeDF.select("make_index").rdd
    //indexedMakeRDD.foreach(y => print(y.toString + " \n"))
    val final_make_RDD = indexedMakeRDD.map(row => row(0))
    //final_make_RDD.foreach(y => print(y.toString + " \n"))


    //------------- MODEL  ------------------//

    val model_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("model_index")
    val modelRDD = allData.map(v => v.model)
    val indexedModelDF = model_indexer.fit(modelRDD.toDF()).transform(modelRDD.toDF())
    //indexedModelDF.show()
    val model_converter = new IndexToString()
      .setInputCol("model_index")
      .setOutputCol("originalCategory")
    val converted_model = model_converter.transform(indexedModelDF)
    //converted_model.select("value", "model_index", "originalCategory").show()
    val indexedModelRDD   = indexedModelDF.select("model_index").rdd
    //indexedModelRDD.foreach(y => print(y.toString + " \n"))
    val final_model_RDD = indexedModelRDD.map(row => row(0))
    //final_model_RDD.foreach(y => print(y.toString + " \n"))



    //------------ LOCATION  ------------------//

    val location_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("location_index")
    val locationRDD = allData.map(v => v.location)
    val indexedLocationDF = location_indexer.fit(locationRDD.toDF()).transform(locationRDD.toDF())
    //indexedLocationDF.show()
    val location_converter = new IndexToString()
      .setInputCol("location_index")
      .setOutputCol("originalCategory")
    val converted_location = location_converter.transform(indexedLocationDF)
    //converted_location.select("value", "location_index", "originalCategory").show()
    val indexedLocationRDD   = indexedLocationDF.select("location_index").rdd
    //indexedLocationRDD.foreach(y => print(y.toString + " \n"))
    val final_location_RDD = indexedLocationRDD.map(row => row(0))
    //final_location_RDD.foreach(y => print(y.toString + " \n"))  // this is not categorial value actually, but I will deal with this later.


    //------------ DRIVER_CITY  ------------------//


    val driver_city_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("driver_city_index")
    val driver_cityRDD = allData.map(v => v.driver_city)
    val indexedDriver_cityDF = driver_city_indexer.fit(driver_cityRDD.toDF()).transform(driver_cityRDD.toDF())
    //indexedDriver_cityDF.show()
    val driver_city_converter = new IndexToString()
      .setInputCol("driver_city_index")
      .setOutputCol("originalCategory")
    val converted_driver_city = driver_city_converter.transform(indexedDriver_cityDF)
    //converted_driver_city.select("value", "driver_city_index", "originalCategory").show()
    val indexedDriver_cityRDD   = indexedDriver_cityDF.select("driver_city_index").rdd
    //indexedDriver_cityRDD.foreach(y => print(y.toString + " \n"))
    val final_driver_city_RDD = indexedDriver_cityRDD.map(row => row(0))
    //final_driver_city_RDD.foreach(y => print(y.toString + " \n"))

    //-------------------- DRIVER_STATE ------------------//

    val driver_state_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("driver_state_index")
    val driver_stateRDD = allData.map(v => v.driver_state)
    val indexedDriver_stateDF = driver_state_indexer.fit(driver_stateRDD.toDF()).transform(driver_stateRDD.toDF())
    //indexedDriver_stateDF.show()
    val driver_state_converter = new IndexToString()
      .setInputCol("driver_state_index")
      .setOutputCol("originalCategory")
    val converted_driver_state = driver_state_converter.transform(indexedDriver_stateDF)
    //converted_driver_state.select("value", "driver_state_index", "originalCategory").show()
    val indexedDriver_stateRDD   = indexedDriver_stateDF.select("driver_state_index").rdd
    //indexedDriver_stateRDD.foreach(y => print(y.toString + " \n"))
    val final_driver_state_RDD = indexedDriver_stateRDD.map(row => row(0))
    //final_driver_state_RDD.foreach(y => print(y.toString + " \n"))


    //-------------------- DL_STATE ------------------//

    val dl_state_indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("dl_state_index")
    val dl_stateRDD = allData.map(v => v.dl_state)
    val indexedDl_stateDF = dl_state_indexer.fit(dl_stateRDD.toDF()).transform(dl_stateRDD.toDF())
    //indexedDl_stateDF.show()
    val dl_state_converter = new IndexToString()
      .setInputCol("dl_state_index")
      .setOutputCol("originalCategory")
    val converted_dl_state = dl_state_converter.transform(indexedDl_stateDF)
    //converted_dl_state.select("value", "dl_state_index", "originalCategory").show()
    val indexedDl_stateRDD   = indexedDl_stateDF.select("dl_state_index").rdd
    //indexedDl_stateRDD.foreach(y => print(y.toString + " \n"))
    val final_dl_state_RDD = indexedDl_stateRDD.map(row => row(0))
    //final_dl_state_RDD.foreach(y => print(y.toString + " \n"))

    //----------------------------------------------------------------------------------------//

    import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
    // NORMALIZING THE DATA
    /*
    var encoder = new OneHotEncoder()
      .setInputCol("dl_state_index")
      .setOutputCol("categoryVec")
    val encoded = encoder.transform(converted_dl_state)
    encoded.show(50)
    */


    // VECTORING TEXT-BASED FEATURES

    val color_doubled : RDD[Double]= final_color_RDD.map(db => (db.toString().toDouble))
    val color_vector = color_doubled.map(d => Vectors.dense(d))
    val location_doubled : RDD[Double]= final_location_RDD.map(db => (db.toString().toDouble))
    val location_vector = location_doubled.map(d => Vectors.dense(d))
    val make_doubled : RDD[Double]= final_make_RDD.map(db => (db.toString().toDouble))
    val make_vector = make_doubled.map(d => Vectors.dense(d))
    val model_doubled : RDD[Double]= final_model_RDD.map(db => (db.toString().toDouble))
    val model_vector = model_doubled.map(d => Vectors.dense(d))
    val driver_city_doubled : RDD[Double]= final_driver_city_RDD.map(db => (db.toString().toDouble))
    val driver_city_vector = driver_city_doubled.map(d => Vectors.dense(d))
    val driver_state_doubled : RDD[Double]= final_driver_state_RDD.map(db => (db.toString().toDouble))
    val driver_state_vector = driver_state_doubled.map(d => Vectors.dense(d))
    val dl_state_doubled : RDD[Double]= final_dl_state_RDD.map(db => (db.toString().toDouble))
    val dl_state_vector = dl_state_doubled.map(d => Vectors.dense(d))

    val text_vectors = color_vector.union(location_vector).union(make_vector).union(model_vector).union(driver_city_vector).union(driver_state_vector).union(dl_state_vector)
    val vectors = allData.map(v => Vectors.dense(stopTime(v.time_of_stop),  vehicleType(v.vehicle_type), validYear(v.year), raceCategory(v.race), genderCategory(v.gender),violationCategory(v.violation_type), boolCategory(v.contributed_to_accident), boolCategory(v.accident),boolCategory(v.belts),boolCategory(v.property_damage),boolCategory(v.fatal),boolCategory(v.commercial_license),boolCategory(v.hazmat),boolCategory(v.commercial_vehicle),boolCategory(v.alcohol),boolCategory(v.work_zone)))
    val allvectors = vectors.union(text_vectors)
    allvectors.cache()




    //------------------------------ CLUSTERING -------------------------//


    val numberOfClusters = 2
    val numberOfIterations = 20
    val kMeansModel = KMeans.train(vectors, numberOfClusters, numberOfIterations)
    println("CLUSTER CENTERS : ")
    kMeansModel.clusterCenters.foreach(println)
    //Vectors.dense( stopTime(v.time_of_stop),  vehicleType(v.vehicle_type), validYear(v.year), raceCategory(v.race), /*genderCategory(v.gender),violationCategory(v.violation_type),*/ boolCategory(v.contributed_to_accident), boolCategory(v.accident),boolCategory(v.belts),boolCategory(v.personal_injury),boolCategory(v.property_damage),boolCategory(v.fatal),boolCategory(v.commercial_license),boolCategory(v.hazmat),boolCategory(v.commercial_vehicle),boolCategory(v.alcohol),boolCategory(v.work_zone))
    val predictions = allData.map(v => (v.race, v.gender, v.vehicle_type, v.make, v.model, kMeansModel.predict(Vectors.dense(stopTime(v.time_of_stop),  vehicleType(v.vehicle_type), validYear(v.year), raceCategory(v.race), genderCategory(v.gender),violationCategory(v.violation_type), boolCategory(v.contributed_to_accident), boolCategory(v.accident),boolCategory(v.belts),boolCategory(v.property_damage),boolCategory(v.fatal),boolCategory(v.commercial_license),boolCategory(v.hazmat),boolCategory(v.commercial_vehicle),boolCategory(v.alcohol),boolCategory(v.work_zone)))))
    val predDF = predictions.toDF("RACE", "GENDER", "VEHICLE TYPE", "MAKE", "MODEL", "CLUSTER")
    //predDF.show(100)


    //predDF.filter("CLUSTER = 1").show(50)
    //println("CLUSTER (0) has " + predDF.filter("CLUSTER = 0").count() + " violations.")
    //println("CLUSTER (1) has " + predDF.filter("CLUSTER = 1").count() + " violations.")







    //------------------------------ CLASSIFICATION -------------------------//


    import org.apache.spark.mllib.tree.RandomForest
    import org.apache.spark.mllib.tree.configuration.Strategy



    // CLASSIFY AS THE VIOLATION CAUSED A PERSONAL INJURY OR NOT:

    /*
    val classif_data = allData.map {v =>
      val label = if (v.personal_injury.equals("Yes")) 1.0 else 0.0
      val vec = Vectors.dense(stopTime(v.time_of_stop),  vehicleType(v.vehicle_type), validYear(v.year), raceCategory(v.race), genderCategory(v.gender),violationCategory(v.violation_type), boolCategory(v.contributed_to_accident), boolCategory(v.accident),boolCategory(v.belts),boolCategory(v.property_damage),boolCategory(v.fatal),boolCategory(v.commercial_license),boolCategory(v.hazmat),boolCategory(v.commercial_vehicle),boolCategory(v.alcohol),boolCategory(v.work_zone))
      LabeledPoint(label, vec)
    }

    val splits = classif_data.randomSplit(Array(0.7,0.3))
    val(training, testing) = (splits(0),splits(1))
    val numberOfTrees = 3
    val seed = 12345

    val model = RandomForest.trainClassifier(training,Strategy.defaultStrategy("Classification"),numberOfTrees,"auto",seed)

    val labelAndPreds = testing.map {point  =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    labelAndPreds.collect().foreach(f =>
      println("Existence of Personal Injury : " + f._1 + "  |  Prediction: " + f._2)
    )

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / (testing.count.toDouble)
    println("Existence of Personal Injury Prediction Success Rate = %" + (100.0 - (testErr*100.0)))
    //println("Learned classification forest model:\n" + model.toDebugString)
    */
    
    //-----------------------------------------//

    // CLASSIFY AS THE GENDER OF THE VIOLATOR

    /*
    // 0: MALE , 1:FEMALE
    val classif_data2 = allData.map {v =>
      val label1 = if (genderCategory(v.gender) == 1) 1.0 else 0.0
      val vec1 = Vectors.dense(stopTime(v.time_of_stop),  vehicleType(v.vehicle_type), validYear(v.year), raceCategory(v.race), violationCategory(v.violation_type), boolCategory(v.contributed_to_accident), boolCategory(v.accident),boolCategory(v.belts),boolCategory(v.property_damage), boolCategory(v.personal_injury), boolCategory(v.fatal),boolCategory(v.commercial_license),boolCategory(v.hazmat),boolCategory(v.commercial_vehicle),boolCategory(v.alcohol),boolCategory(v.work_zone))
      LabeledPoint(label1, vec1)
    }
    val splits2 = classif_data2.randomSplit(Array(0.7,0.3))
    val(training2, testing2) = (splits2(0),splits2(1))
    val numberOfTrees2 = 3
    val seed2 = 12345

    val model2 = RandomForest.trainClassifier(training2,Strategy.defaultStrategy("Classification"),numberOfTrees2,"auto",seed2)

    val labelAndPreds2 = testing2.map {point  =>
      val prediction2 = model2.predict(point.features)
      (point.label, prediction2)
    }

    labelAndPreds2.collect().foreach(f =>
      println("Real Gender : " + indexToGender(f._1) + "  |  Predicted Gender: " + indexToGender(f._2))
    )

    val testErr2 = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / (testing2.count.toDouble)
    println("Gender Prediction Success Rate = %" + (100.0 - (testErr2*100.0)))
    //println("Learned classification forest model:\n" + model.toDebugString)
    */





  } // end of main function






  def remove(a: Array[String], index: Int): Array[String] = {
    val b = a.toBuffer
    b.remove(index)
    b.toArray
  }

  def vehicleType(vehicleString : String) : Int = {
    var veh_type = -1
    val givenType = vehicleString.substring(0,2) // first two characters give vehicle code
    if(givenType.charAt(0) == '0' || givenType.charAt(0) == '1' || givenType.charAt(0) == '2'){ // from 0 to 29 different codes
      veh_type = givenType.toInt
    }
    veh_type
  }

  def validYear(year : String) : Int = {
    var result = -1
    if((year.length == 4) && ((year.charAt(0) == '1' || year.charAt(0) == '2'))){
      result = year.toInt
      if(result < 1940 || result > 2017){
        result = 2001
      }
    }
    else{
      result = 2001
    }
    return result
  }

  def stopTime(time : String) : Int = {
    var day_time = -1
    val hour = time.substring(0,2).toInt // format: hh:mm:ss
    //print("hour is : " + hour)
    if(hour >= 0 && hour < 6 ){ // night
      day_time = 0
    }
    else if(hour >= 6 && hour < 12 ){ // morning
      day_time = 1
    }
    else if(hour >= 12 && hour < 18 ){ // noon
      day_time = 2
    }
    else if(hour >= 18 && hour < 24 ){ // afternoon
      day_time = 2
    }
    //print("day time is : " + day_time)
    day_time
  }

  def printUniqueElements(rddx : RDD[String]): Unit = {
    val uniques = rddx.distinct().collect()
    for(unique <- uniques){
      print("|" +unique + "| \n")
    }
  }

  def printUniqueElements_int(rddx : RDD[Int]): Unit = {
    val uniques = rddx.distinct().collect()
    for(unique <- uniques){
      print("|" +unique + "| \n")
    }
  }

  def state_number(states : Array[String], state : String) : Int = {
    var sorted_uniques = remove(states, states.indexOf(" ")).sorted
    val index = sorted_uniques.indexOf(state)
    print("index : " + index)
    index
  }

  def state_codes(states : Array[String]) : Array[Int] = {
    var sorted_uniques = states.sorted
    var res = new ListBuffer[Int]()
    for(state <- states){
      var a = -1
      if(sorted_uniques contains(state)){
        a = sorted_uniques.indexOf(state)
      }
      else{
        a = -1
      }
      res += a
    }
    return res.toArray
  }

  def printArray(arr : Array[String]) : Unit = {
    for(elem <- arr){
      print(elem + " \n")
    }
  }

  def stateStatus(state: String) : Int = {
    var result = -1
    if(state.equals("MD")){
      result = 0    // state is MD
    }
    else{
      result = 1    // other states
    }
    result
  }

  def raceCategory(race : String) : Int = {
    var category = -1
    if(race.equals("WHITE")){
       category = 0
    }
    else if(race.equals("ASIAN")){
       category = 1
    }
    else if(race.equals("BLACK")){
       category = 2
    }
    else if(race.equals("OTHER")){
       category = 3
    }
    else if(race.equals("No")){
       category = 4
    }
    else if(race.equals("NATIVE AMERICAN")){
       category = 5
    }
    else if(race.equals("HISPANIC")){
       category = 6
    }
    else{
       category = -1
    }
    category
  }

  def genderCategory(gender : String) : Int = {
    var result = -1
    if(gender.equals("M")){
      result = 0
    }
    else if(gender.equals("F")){
      result = 1
    }
    else{
      print("")
    }
    result
  }

  def indexToGender(index : Double) : String = {
    var gender = "Male"
    if(index == 1.0){
      gender = "Female"
    }
    gender
  }

  def violationCategory(violationType : String) : Int = {
    var result = -1
    if(violationType.equals("Citation")){
      result = 0
    }
    else if(violationType.equals("Warning")){
      result = 1
    }
    else{
      print("")
    }
    result
  }

  def boolCategory(answer : String) : Int = {
    var result = -1
    if(answer.equals("Yes")){
      result = 1
    }
    else if(answer.equals("No")){
      result = 0
    }
    else{
      print("")
    }
    result
  }

}

