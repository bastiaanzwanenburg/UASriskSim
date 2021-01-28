class riskDatabase():
    '''
    Database in which the risk of a specific crash can be stored. This greatly increases computational efficiency.
    '''
    def __init__(self, loadDatabaseFromFile=False, rounding_factor=1):
        self.database = dict()
        self.rf = rounding_factor  # Factor by which we round the pos and v_horizontal before entering them in the array

        if loadDatabaseFromFile:
            raise Exception("This function is not included yet")

    def generate_hash(self, position, speed):
        '''A hash is generated based on position & speed to store and retrieve a risk entry.'''
        if type(position) == tuple and type(speed) == tuple:
            position = [round(x, self.rf) for x in position]
            speed = [round(x, self.rf) for x in speed]

            hash = str(position) + "/" + str(speed)
            return hash
        else:
            raise Exception("Storing path in database but either origin or target is not tuple")

    def decode_hash(self, hash):
        '''A hash is decoded to arrive at the position and speed corresponding to it.'''
        position = eval(hash.split("/")[0])
        speed = eval(hash.split("/")[1])
        return position, speed

    def add_risk_to_database(self, position, speed, risk_map, store_to_file=False):
        '''Method for adding the risk to the database.'''
        if type(position) == tuple and type(speed) == tuple:
            hash = self.generate_hash(position, speed)
            if hash in self.database:
                raise Exception(
                    "adding risk for a pos/v_horizontal combo that is already in the database - that shouldn't be possible")
            if type(risk_map) == float:
                self.database[hash] = risk_map
            else:
                self.database[hash] = risk_map[:]  # The [:] is needed because otherwise the database entry is linked to the list.
        else:
            raise Exception("Storing path where either origin or target is not a tuple.")

        if store_to_file:
            with open("riskCalculations.txt", "wb") as myFile:
                pickle.dump(self.database, myFile)

    def retrieve_risk(self, position, speed):
        '''Retrieves risk from the database.'''
        # Either returns the path, otherwise returns 0.
        hash = self.generate_hash(position, speed)
        if hash in self.database:
            risk_map = self.database.get(hash)
            return risk_map
        else:
            return None

class crashLocationDatabase():
    '''Database for storing the crash-locations following a failure at a location.'''
    def __init__(self, loadDatabaseFromFile=False, rounding_factor_speed=1, rounding_factor_altitude=0):
        self.locations_database = dict()
        self.E_impact_database = dict()
        self.vimp_database = dict()
        self.rf_speed = rounding_factor_speed  # Factor by which we round the pos and v_horizontal before entering them in the array
        self.rf_altitude = rounding_factor_altitude
        if loadDatabaseFromFile:
            raise Exception("This function is not included yet")

    def generate_hash(self, speed, altitude):
        ''' Generate hash based on speed and altitude '''
        if type(speed) == tuple and len(speed) == 3:
            # position = [round(x, self.rf) for x in position]
            speed = [round(x, self.rf_speed) for x in speed]
            altitude = round(altitude, self.rf_altitude)
            hash = str(altitude) + "/" + str(speed)
            return hash
        else:
            raise Exception("Storing crash locations in database but speed is not tuple")

    # This function is not needed (as of now)
    # def decode_hash(self, hash):
    #     position = eval(hash.split("/")[0])
    #     speed = eval(hash.split("/")[1])
    #     return position, speed

    def add_crash_locations_to_database(self, speed, altitude, crash_locations, E_impact_locations, vimp):
        ''' Add crash locations to the database '''
        if type(crash_locations) == dict and type(E_impact_locations) == dict and type(vimp) == dict and type(speed) == tuple and len(speed) == 3:
            hash = self.generate_hash(speed, altitude)
            if hash in self.locations_database or hash in self.E_impact_database:
                raise Exception("Adding crash locations for a position that is already in the database")
            else:
                self.locations_database[hash] = crash_locations
                self.E_impact_database[hash] = E_impact_locations
                self.vimp_database[hash] = vimp
        else:
            raise Exception("Error here!")

    def retrieve_crash_locations(self, speed, altitude):
        ''' Retrieve crash locations from the database '''
        if type(speed) == tuple:
            hash = self.generate_hash(speed, altitude)
            if hash in self.locations_database and hash in self.E_impact_database:
                location_database = self.locations_database.get(hash)
                E_impact_locations = self.E_impact_database.get(hash)
                v_impact_database = self.vimp_database.get(hash)
                return location_database, E_impact_locations, v_impact_database
            else:
                return None, None, None





class routeDatabase():
    '''Stores paths in the database.'''
    def __init__(self, db_url, lock_url, symmetric=False, loadDatabaseFromFile=False):
        '''

        @param db_url: link to the database file.
        @param lock_url: link to a lock-file. This is a bit hacky - the lock-file isn't used by itself. However, whenever the lock-file is locked, the program will wait with writing to the db to avoid that the DB becomes corrupt because several processes write to it simultaneously.
        @param symmetric: whether or not path from A --> B should be the same as path from B --> A.
        @param loadDatabaseFromFile: whether or not a database should be pre-loaded from the file.
        '''

        self.database = dict()
        self.symmetric = symmetric

        self.db_url = db_url
        self.lock_url = lock_url

        self.loadDatabaseFromFile = loadDatabaseFromFile

        if loadDatabaseFromFile:
            print("loading database")
            try:
                with open("savedRoutes.txt", "rb") as myFile:
                    self.database = pickle.load(myFile)
            except IOError:
                print("Error loading route database from file")

    def generate_hash(self, origin, target):
        ''' Generate hash based on origin and target positions.
        @return hash'''
        if type(origin) == tuple and type(target) == tuple:
            hash = str(origin) + "/" + str(target)
            return hash
        else:
            raise Exception("Storing path in database but either origin or target is not tuple")

    def decode_hash(self, hash):
        '''Decode a hash
        @return origin & target '''
        origin = eval(hash.split("/")[0])
        target = eval(hash.split("/")[1])
        return origin, target

    def add_to_database(self, origin, target, path, store_to_file=False):
        '''

        @param origin: origin position
        @param target: target position
        @param path: path between origin and target
        @param store_to_file: whether or not to store the db to a file
        @return: None
        '''
        if type(origin) == tuple and type(target) == tuple:
            hash = self.generate_hash(origin, target)
            if hash in self.database:
                raise Exception("adding a path that is already in the database - that shouldn't be possible")
            self.database[hash] = path[:]  # The [:] is needed because otherwise the database entry is linked to the list.
        else:
            raise Exception("Storing path where either origin or target is not a tuple.")

        if self.symmetric:
            # If symmetric is on, also add the reverse path.
            hash = self.generate_hash(target, origin)
            self.database[hash] = path[:]

        if store_to_file == True:
            with portalocker.Lock(self.lock_url, "w") as lockFile:
                with portalocker.Lock(self.db_url, "rb") as databaseFile:
                    # Load database
                    db = pickle.load(databaseFile)

                with portalocker.Lock(self.db_url, "wb") as databaseFile:
                    if type(origin) == tuple and type(target) == tuple:
                        hash = self.generate_hash(origin, target)
                        db[hash] = path[:]
                        if self.symmetric:
                            hash2 = self.generate_hash(target, origin)
                            db[hash2] = path[:]
                    pickle.dump(db, databaseFile)

            # with open("savedRoutes.txt", "wb") as myFile:
            #     pickle.dump(self.database, myFile)

    def retrieve_path(self, origin, target):
        '''
        @param origin: origin position
        @param target: target position
        @return: path, or None if no path is in the db.
        '''
        # If paths are generated randomly, retrieving path-data from the database-file isn't very efficient.

        # Either returns the path, otherwise returns 0.
        hash = self.generate_hash(origin, target)
        if hash in self.database:
            return_path = self.database.get(hash)
            return return_path[:]
        else:
            # check if it is in the file database
            if self.loadDatabaseFromFile:
                if os.path.getsize(self.db_url) > 0:
                    with portalocker.Lock(self.db_url, "rb") as databaseFile:
                        # Load database
                        self.database_temp = pickle.load(databaseFile)
                    if hash in self.database_temp:
                        print("Found the path in the file!")
                        return_path = self.database.get(hash)
                        return return_path[:]
                    else:
                        return None
            else:
                return None


