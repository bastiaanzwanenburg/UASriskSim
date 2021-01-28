'''
Loads the environment from images of sheltering data and population density data. Also includes methods for modifying the population density and scaling the environment.
'''

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# from env.RiskMap2 import shelter_matrix
import math
from skimage.util import view_as_blocks
from scipy import stats
import scipy.ndimage
from warnings import warn

cities = {"Delft":
                # total population = 67868 people
              {"url_pop": "env/city_inputs/delft/delftdichtheid2.png",
               "url_shelter": "env/city_inputs/delft/delft_v3_edited.png", # Make sure that the same file is used here as in the __main__!!!!
               "url_modifyPopDens": "env/city_inputs/delft/modify_pop_dens.png",
               "width_meters": 4600,
               "height_meters": 3800,
               },
          "UniformTown":
              {"url_pop": None,
               "url_shelter": None,
               "url_modifyPopDens": None,
               "width_meters": 4600,
               "height_meters": 3800},
          "NewTown":
              {"url_pop": None,
               "url_shelter": "env/city_inputs/newtown/newtown.png",
               "url_modifyPopDens": None,
               "width_meters": 1000,
               "height_meters": 1000},
          "NewYork":
          # total population = 1.26m people
              {"url_pop": "env/city_inputs/newyork/NY_popdens_v3.png",
               "url_modifyPopDens": "env/city_inputs/newyork/office_or_home.png",
               "url_shelter": "env/city_inputs/newyork/ny_corrected_map.png",
                "width_meters": 9300,
               "height_meters": 8700
               },
          "Paris":
          # total popoulation = 3.84m people
              {"url_pop": "env/city_inputs/paris/paris_dens_map_v3.png",
               "url_modifyPopDens": "",
               "url_shelter": "env/city_inputs/paris/paris_output_v2.png",
               "width_meters": 12000,
               "height_meters": 10400
               },
          "CityGrid":
              {"url_pop": None,
               "url_shelter": None,
               "url_modifyPopDens": None,
               "width_meters": 1000,
               "height_meters": 1000}
          }

# if __name__ == "__main__":
    #
    # new_shelter_url = "city_inputs/delft/delft_v3_edited.png"
    # new_popdens_url = "city_inputs/delft/delftdichtheid2.png"
    # new_modifypopdens_url = "city_inputs/delft/modify_pop_dens.png"
    # if ("env/" + new_shelter_url) != cities["Delft"]["url_shelter"]:
    #     raise Exception("shelter_matrix urls are not matching")
    # if ("env/" + new_popdens_url) != cities["Delft"]["url_pop"]:
    #     raise Exception("Popdens urls are not matching")
    # if ("env/" + new_modifypopdens_url) != cities["Delft"]["url_modifyPopDens"]:
    #     raise Exception("Modifypopdens urls are not matching")
    #
    # cities["Delft"]["url_pop"] = new_popdens_url
    # cities["Delft"]["url_shelter"] = new_shelter_url
    # cities["Delft"]["url_modifyPopDens"] = new_modifypopdens_url
    # cities["NewTown"]["url_shelter"] = "city_inputs/newtown/newtown.png"
    # import matplotlib.pyplot as plt


class loadEnvironment():
    def __init__(self, city="Delft", pop_dens_scenario=None, scaling_factor=1, risk_map_resolution=1, water_setting = 0, modify_dens_list = []):
        """
        The loadEnvironment class, which loads the cities
        city: Chooses the city to load
        scaling_factor: the factor by which all regular maps are scaled (this is used for path-finding)
        risk_map_resolution: the factor by which all _risk maps are scaled (these are used for risk-computation)
        water_setting: whether or not to remove all pop dens from water (0 = no pop dens on water)
        modify_dens_list: accepts a list which can be used to modify pop density

        """
        self.purge_cache = True
        self.waterSetting = water_setting
        self.url_pop = cities[city]["url_pop"]
        self.url_shelter = cities[city]["url_shelter"]
        self.url_modifyPopDens = cities[city]["url_modifyPopDens"]

        self.scaling_factor_normal = scaling_factor
        self.scaling_factor_risk = risk_map_resolution

        self.city = city

        self.width_meters = cities[city]["width_meters"]
        self.height_meters = cities[city]["height_meters"]

        # These two functions load all pop-, and sheltering-data
        self.loadPopulationDensity()
        self.loadShelteringData()
        self.shelter_matrix[self.shelter_matrix == 0] = 0.001


        self.shelter_category_names = np.append(self.shelter_category_names, "hub")
        self.shelter_category_names = np.append(self.shelter_category_names, "delivery point")


        # A function which has some statistics on how big the area of each category is.
        self.area_per_shelter_category = np.zeros((len(self.shelter_category_names)))
        for i in range(len(self.area_per_shelter_category)):
            self.area_per_shelter_category[i] = np.count_nonzero(self.shelter_category == i)


        # if pop_dens_scenario == "working_hours":
        #     if self.city != "Delft":
        #         raise Exception("Modifying pop density only supported for delft")
        #
        #     self.modifyPopulationDensity(target_home=0.0, target_office=(0.0), target_transport=1.0, target_other=0.0)
        #     self.density_matrix = self.densitymatrix_modified.copy()

        if water_setting == 1:
            self.modifyPopDensityWater()
        elif water_setting == 0 :
            self.removePopDensityFromWater() # TODO: This is undone by modifying pop density
        # else:
        #     print("WARNING: not doing anything with water as city is NY")


        # Usually: ['Error' 'Office' 'Water' 'Street' 'Event Location' 'Home Area', 'Small Street' 'Meadow']
        if len(modify_dens_list) > 0:
            self.modifyPopulationDensity(modify_dens_list) # Hub & Deliverypoint are only appended later so therefore this is the correct size of the array.
            self.density_matrix = self.density_matrix_modified.copy()



        # In the following, all maps are scaled. For density-maps, "average" scaling has shown to be optimal,
        # while for sheltering, "mode" has shown to be optimal
        self.density_matrix_scaled = self.scaleMap(self.density_matrix, scaling_factor=self.scaling_factor_normal,
                                                   scaling_method="average", file_name="density_matrix_scaled")
        self.shelter_map_scaled = self.scaleMap(self.shelter_matrix, scaling_factor=self.scaling_factor_normal,
                                                scaling_method="mode", file_name="shelter_map_scaled")

        self.density_matrix_scaled_risk = self.scaleMap(self.density_matrix, scaling_factor=self.scaling_factor_risk,
                                                        scaling_method="average", file_name="density_matrix_scaled_risk")
        self.shelter_map_scaled_risk = self.scaleMap(self.shelter_matrix, scaling_factor=self.scaling_factor_risk,
                                                     scaling_method="mode", file_name="shelter_map_scaled_risk")

        self.shelter_category = self.shelter_category
        self.shelter_category_scaled = self.scaleMap(self.shelter_category, scaling_factor=self.scaling_factor_normal,
                                                           scaling_method="mode", file_name="shelter_category_scaled")  # TODO scale this appropriately for both risk and pathfinding

        self.shelter_category_scaled_risk = self.scaleMap(self.shelter_category, scaling_factor=self.scaling_factor_risk,
                                                                scaling_method="mode", file_name="shelter_category_scaled_risk")




        # Extract the parameters of the maps (sizes etc)
        self.Width = self.shelter_map_scaled.shape[0]
        self.Height = self.shelter_map_scaled.shape[1]
        self.width_risk = self.density_matrix_scaled_risk.shape[0]
        self.height_risk = self.density_matrix_scaled_risk.shape[1]

        self.width_risk_m = self.width_meters / self.width_risk
        self.height_risk_m = self.height_meters / self.height_risk

        self.risk_grid_size_m_x = self.width_meters / self.density_matrix_scaled_risk.shape[0]
        self.risk_grid_size_m_y = self.height_meters / self.density_matrix_scaled_risk.shape[1]

        self.grid_size_m_x = self.width_meters / self.density_matrix_scaled.shape[0]
        self.grid_size_m_y = self.height_meters / self.density_matrix_scaled.shape[1]

        self.area_risk_m = self.width_risk_m * self.height_risk_m

        self.statisticsPopulationDensity(verbose=False)  # This function is required for some other things

        if (self.density_matrix_scaled.shape != self.shelter_map_scaled.shape) or (self.density_matrix.shape != self.shelter_matrix.shape) or (
                self.density_matrix_scaled_risk.shape != self.shelter_map_scaled_risk.shape) or (self.shelter_category_scaled_risk.shape != self.density_matrix_scaled_risk.shape) or (self.shelter_category_scaled.shape != self.shelter_map_scaled.shape):
            raise Exception("Dimensions of density matrix and sheltering do not match")



        if self.density_matrix.shape != self.shelter_matrix.shape:
            raise Exception("Dimensions of Density Matrix and Sheltering Map differ")

        # elif self.densitymatrix_scaled.shape != self.Shelter_scaled.shape:
        #     print(self.densitymatrix_scaled.shape)
        #     print(self.Shelter_scaled.shape)
        #     # raise Exception("Dimensions of scaled Density Matrix and Sheltering Map differ")

    def loadPopulationDensity(self):
        '''
        Method that is ran whenever the loadEnvironment class is initialized. It loads the population density of the current city.
        '''
        if self.city == "CityGrid":
            return
        if self.url_pop == None:
            if self.city == "UniformTown":
                self.density_matrix = np.ones((909, 749))
            elif self.city == "NewTown":
                self.density_matrix = np.ones((38, 38))
        elif self.city == "Delft":
            try:
                if self.purge_cache:
                    raise FileNotFoundError
                self.density_matrix = np.load("env/database/delft/density_matrix.npy")
                #print("loading from db")

            except FileNotFoundError:
                picca2 = Image.open(self.url_pop)
                picca2 = picca2.convert('RGB')
                self.width, self.height = picca2.size  # obtaining dimensions of figure

                self.density_matrix = np.zeros((self.width, self.height))  # setting up the numpy array
                for i in range(0, self.width):
                    for j in range(0, self.height):

                        coordinates = i, j  # creating tuple as input of the next line
                        red, green, blue = picca2.getpixel(coordinates)  # obtaining the values of RGB for every pixel

                        # Pop density = pop/m^2, source = https://delft.incijfers.nl/jive?cat_show=Bevolking
                        if red > 230 and green > 230 and blue > 230:  # assigning a value of 0 to all areas with no population density
                            self.density_matrix[i, j] = 0.01 / 100

                        elif red == 255 and green > 195 and blue > 150:  # assigning a value of 1 to all areas based on their population density
                            self.density_matrix[i, j] = 0.1 / 100

                        elif red == 244 and green == 156 and blue > 5:  # assigning a value of 2 to all areas based on their population density
                            self.density_matrix[i, j] = 0.7 / 100

                        elif red == 231 and green == 77 and blue > 20:  # assigning a value of 3 to all areas based on their population density
                            self.density_matrix[i, j] = 0.012

                        elif red == 192 and green == 31 and blue > 35:  # assigning a value of 4 to all areas based on their population density
                            self.density_matrix[i, j] = 0.016

                        elif red == 130 and green >= 0 and blue == 30:  # assigning a value of 5 to all areas based on their population density
                            self.density_matrix[i, j] = 0.023 #actual value: 23k citizens / km^2 (or 0.023 citizens per m^2)

                        else:  # assigning a cost of 0 to pixels that meet none of these if statements
                            self.density_matrix[i, j] = 0.01 / 100
                            self.density_matrix[i, j] = self.density_matrix[i][j - 1]

                #np.save("env/database/delft/density_matrix.npy", self.density_matrix)

            # self.density_matrix = self.density_matrix.transpose()
        elif self.city == "NewYork":
            try:
                if self.purge_cache:
                    raise FileNotFoundError
                self.density_matrix = np.load("env/database/ny/density_matrix.npy")
                #print("loading from db")

            except FileNotFoundError:
                picca2 = Image.open(self.url_pop)
                picca2 = picca2.convert('RGB')
                self.width, self.height = picca2.size  # obtaining dimensions of figure

                self.density_matrix = np.zeros((self.width, self.height))  # setting up the numpy array
                for i in range(0, self.width):
                    for j in range(0, self.height):

                        coordinates = i, j  # creating tuple as input of the next line
                        red, green, blue = picca2.getpixel(coordinates)  # obtaining the values of RGB for every pixel

                        # Pop density = pop/m^2, source = https://delft.incijfers.nl/jive?cat_show=Bevolking
                        if red == 230 and green == 223 and blue == 20:  # Yellow, 30,000 - 150,000 ppl / square km
                            self.density_matrix[i, j] = 0.09

                        elif red == 93 and green == 69 and blue == 71:  # dark dark red (lowest ppl dens, 1,500 - 3,330 ppl / sq km)
                            self.density_matrix[i, j] = (1500 + 3300) / (2*1e6)

                        elif red == 167 and green == 28 and blue == 38:  # (dark)-red (3,330 - 7,000)
                            self.density_matrix[i, j] = (3330 + 7000) / (2*1e6)

                        elif red == 227 and green == 143 and blue == 64:  # orange, (7,000 - 30,000) ppl / sq KM
                            self.density_matrix[i, j] = (7000 + 30000)  / (2*1e6)

                        elif red == 35 and green == 34 and blue == 39:  # assigning a value of 4 to all areas based on their population density
                            self.density_matrix[i, j] = 0.0

                        elif red == 96 and green == 96 and blue == 99:  # assigning a value of 5 to all areas based on their population density
                            self.density_matrix[i, j] = 0.0 #actual value: 23k citizens / km^2 (or 0.023 citizens per m^2)

                        else:  # If a color is not matched, throw a warning, and take the average of the four surrounding points (
                            print("WARNING: Unmatched color in NY, color = ({}, {}, {})".format(red, green, blue))
                            self.density_matrix[i, j] = 0.25 * (self.density_matrix[i][j + 1] + self.density_matrix[i][j - 1] + self.density_matrix[i-1][j] + self.density_matrix[i+1][j])

                #np.save("env/database/newyork/density_matrix.npy", self.density_matrix)

        elif self.city == "Paris":
            try:
                if self.purge_cache:
                    raise FileNotFoundError
                self.density_matrix = np.load("env/database/paris/density_matrix.npy")
                #print("loading from db")

            except FileNotFoundError:
                picca2 = Image.open(self.url_pop)
                picca2 = picca2.convert('RGB')
                self.width, self.height = picca2.size  # obtaining dimensions of figure

                self.density_matrix = np.zeros((self.width, self.height))  # setting up the numpy array
                for i in range(0, self.width):
                    for j in range(0, self.height):

                        coordinates = i, j  # creating tuple as input of the next line
                        red, green, blue = picca2.getpixel(coordinates)  # obtaining the values of RGB for every pixel

                        # Pop density = pop/m^2, source = https://delft.incijfers.nl/jive?cat_show=Bevolking
                        if red == 142 and green == 29 and blue == 24: #Dark-red = > 50,000 per km^2 #8E1D18 (142, 29, 24)

                            self.density_matrix[i, j] = 50000 / 1e6

                        elif red == 197 and green == 50 and blue == 45:  # Red = 25,000 - 50,000 per km^2 #C5322D (197, 50, 45)
                            self.density_matrix[i, j] = (3*25000 + 50000) / (4*1e6)

                        elif red == 221 and green == 100 and blue == 73:  # Orange = 10,000-25,000 per km^2 #DD6449 (221, 100, 73)
                            self.density_matrix[i, j] = (3*10000 + 25000) / (4*1e6)

                        elif red == 223 and green == 140 and blue == 112:  # Semi-orange = 5,000 - 10,000 per km^2 #DF8C70 (223, 140, 112)
                            self.density_matrix[i, j] = (3*5000 + 10000)  / (4*1e6)

                        elif red == 206 and green == 212 and blue == 213:  # White = <2,500 per km^2 #CED4D5 (206, 212, 213)
                            self.density_matrix[i, j] = (0 + 2500) / (2*1e6)

                        elif red == 95 and green == 144 and blue == 180:  # White = <2,500 per km^2 #CED4D5 (206, 212, 213)
                            self.density_matrix[i, j] = 0.0

                        # elif red == 96 and green == 96 and blue == 99:  # assigning a value of 5 to all areas based on their population density
                        #     self.density_matrix[i, j] = 0.0 #actual value: 23k citizens / km^2 (or 0.023 citizens per m^2)

                        else:  # If a color is not matched, throw a warning, and take the average of the four surrounding points (
                            print("WARNING: Unmatched color in Paris, color = ({}, {}, {})".format(red, green, blue))
                            self.density_matrix[i, j] = 0.25 * (self.density_matrix[i][j + 1] + self.density_matrix[i][j - 1] + self.density_matrix[i-1][j] + self.density_matrix[i+1][j])

                #np.save("env/database/paris/density_matrix.npy", self.density_matrix)

        else:
            raise Exception("Unknown city")


    def loadShelteringData(self):
        '''
        Method that is ran whenever the loadEnvironment class is initialized. It loads the sheltering data of the current city.
        '''
        if self.city == "CityGrid":
            self.generateCityGrid(block_width = 4, block_height=4, street_width=2, n_blocks_width=5, n_blocks_height=5, position_river="middle", width_river=2, n_block_river=2 )
            return

        if self.url_shelter == None:
            self.shelter_matrix = np.ones((909, 749))
            self.shelter_category = np.zeros((909, 749), dtype=int)
            self.shelter_category_names = np.array(["Home Area"])
        else:
            picca = Image.open(self.url_shelter)
            picca = picca.convert('RGB')
            width, height = picca.size  # obtaining dimensions of figure
            self.width = width
            self.height = height

            shelter_factor = np.zeros((width, height))  # setting up the numpy array
            self.isWater = np.zeros((width,height))
            self.shelter_category = np.zeros((width, height), dtype=int)
            unique_colors = []
            if self.city == "NewTown":
                self.shelter_category_names = np.array(
                    ["Error", "Office", "Water", "Street", "Event Location", "Home Area", "Small Street", "Meadow"])

                for i in range(0,width):
                    for j in range(0,height):
                        coordinates = i, j  # creating tuple as input of the next line
                        red, green, blue = picca.getpixel(coordinates)  # obtaining the values of RGB for every pixel

                        if red == 0 and green == 128 and blue == 0:
                            # A green pixel is a park / meadow so has 0 sheltering
                            self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Meadow")[0][0])
                            shelter_factor[i,j] = 0.001 # to avoid divide by zero errors
                        elif red == 0 and green == 0 and blue == 255:
                            # Blue = urban area, so 2.5 sheltering
                            self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Home Area")[0][0])
                            shelter_factor[i,j] = 5
                        elif red == 0 and green == 0 and blue == 0:
                            self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Street")[0][0])
                            # Road, so 1.5 sheltering
                            shelter_factor[i,j] = 2
                        elif red == 255 and green == 0 and blue == 0:
                            self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Office")[0][0])
                            shelter_factor[i,j] = 7.5
                        else:
                            print("Color not matched is {}".format((red, green, blue)))
                            raise Exception("One color is not matched")

            # For new delft
            # pink = (204, 57, 209) = office
            # blue = (19,0,245)     = water
            # yellow = (255,255,0) = street
            # red = (234, 51, 35)   = event location
            # black = (0,0,0)       = home area
            # white = (255,255,255) = small street
            # green = (55, 126, 34) = meadow


            elif self.city == "Delft":
                self.shelter_category_names = np.array(
                    ["Error", "Office", "Water", "Street", "Event Location", "Home Area", "Small Street", "Meadow"])
                shelter_cat_url = "env/database/delft/shelter_category_waterSetting=" + str(self.waterSetting) + ".npy"
                isWater_url = "env/database/delft/is_water_waterSetting=" + str(self.waterSetting) + ".npy"
                shelterMatrixUrl = "env/database/delft/shelter_matrix_waterSetting=" + str(self.waterSetting) + ".npy"
                try:
                    if self.purge_cache:
                        raise FileNotFoundError
                    #print("loading from db")

                    self.shelter_category = np.load(shelter_cat_url)
                    self.isWater = np.load(isWater_url) # This is redundant!
                    self.shelter_matrix = np.load(shelterMatrixUrl)
                except FileNotFoundError:
                    for i in range(0, width):
                        for j in range(0, height):

                            coordinates = i, j  # creating tuple as input of the next line
                            red, green, blue = picca.getpixel(coordinates)  # obtaining the values of RGB for every pixel
                            color = (red,green,blue)

                            if red == 204 and green == 57 and blue == 209:  # Pink, office
                                shelter_factor[i, j] = 7.5
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Office")[0][0])
                            elif red == 55 and green == 126 and blue == 34:
                                shelter_factor[i,j] = 0
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Meadow")[0][0])
                            elif red == 19 and green == 0 and blue == 245:  # blue, water
                                shelter_factor[i, j] = 0
                                self.isWater[i,j] = 1
                                # print("categorising water at position {}".format((i,j)))
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Water")[0][0])
                            elif red == 255 and green == 255 and blue == 0:  # yellow, street
                                shelter_factor[i, j] = 3.5
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Street")[0][0])
                            elif red == 234 and green == 51 and blue == 35: # red, event location
                                shelter_factor[i,j] = 0
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Event Location")[0][0])
                            elif red == 0 and green == 0 and blue == 0: # black, home location
                                shelter_factor[i,j] = 5
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Home Area")[0][0])
                            elif red == 255 and green == 255 and blue == 255:
                                shelter_factor[i,j] = 2.5 # white, small roads
                                self.shelter_category[i, j] = int(np.where(self.shelter_category_names == "Small Street")[0][0])
                            else:
                                print(picca.getpixel(coordinates))
                                raise Exception("Pixel found in city Delft that can't be defined.")

                    if np.count_nonzero(self.shelter_category == 0) > 0:
                            raise Exception("Not all gridpoints have been categorised")

                    # self.shelter_matrix = shelter_factor.transpose()  # for some reason, the array is sideways, here it gets transposed to the correct orientation
                    self.shelter_matrix = shelter_factor
                    self.shelter_matrix[self.shelter_matrix == 0] = 0.001

                    #np.save(shelter_cat_url, self.shelter_category)
                    #np.save(shelterMatrixUrl, self.shelter_matrix)
                    #np.save(isWater_url, self.isWater)

            elif self.city == "NewYork":
                changedensmap = Image.open(self.url_modifyPopDens)
                changedensmap = changedensmap.convert("RGB")


                # 170, 218, 255 = blue = water
                # 199, 233, 199 = green = parks
                # 255, 234, 160 = large roads = yellow
                # 255, 255, 255 = white = roads
                # 232, 232, 232 = grey = residential
                # 254, 247, 224 = light yellow (office districts)
                # 252, 232, 230 = pink, hospital

                self.shelter_category_names = np.array(
                    ["Error", "Office", "Water", "Street", "Event Location", "Home Area", "Small Street", "Meadow"])
                shelter_cat_url = "env/database/newyork/shelter_category_waterSetting=" + str(self.waterSetting) + ".npy"
                isWater_url = "env/database/newyork/is_water_waterSetting=" + str(self.waterSetting) + ".npy"
                shelterMatrixUrl = "env/database/newyork/shelter_matrix_waterSetting=" + str(self.waterSetting) + ".npy"
                try:
                    if self.purge_cache:
                        raise FileNotFoundError
                    #print("loading from db")
                    self.shelter_category = np.load(shelter_cat_url)
                    self.isWater = np.load(isWater_url)  # This is redundant!
                    self.shelter_matrix = np.load(shelterMatrixUrl)
                except FileNotFoundError:
                    for i in range(0, width):
                        for j in range(0, height):
                            coordinates = i, j  # creating tuple as input of the next line
                            red, green, blue = picca.getpixel(
                                coordinates)  # obtaining the values of RGB for every pixel

                            red_cd, green_cd, blue_cd = changedensmap.getpixel(coordinates)

                            if red == 170 and green == 218 and blue == 255:  # blue, water
                                shelter_factor[i, j] = 0
                                self.isWater[i, j] = 1
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Water")[0][0])
                            elif red == 199 and green == 233 and blue == 199: # green, parks
                                shelter_factor[i, j] = 1
                                self.density_matrix[i, j] = 0
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Meadow")[0][0])
                            elif red == 255 and green == 234 and blue == 160:  # yellow
                                shelter_factor[i, j] = 2.5
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Street")[0][0])
                            elif red == 255 and green == 255 and blue == 255:  # white, small street
                                shelter_factor[i, j] = 2.5
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Small Street")[0][0])

                            elif (red == 253 and green == 246 and blue == 224) or (red == 232 and green == 232 and blue == 232) or (red == 252 and green == 232 and blue == 230):  # home / office area
                                if red_cd == 0 and green_cd == 0 and blue_cd == 0:
                                    shelter_factor[i, j] = 7.5 #offices
                                    self.shelter_category[i, j] = int(
                                        np.where(self.shelter_category_names == "Office")[0][0])
                                else:
                                    shelter_factor[i, j] = 5
                                    self.shelter_category[i, j] = int(
                                        np.where(self.shelter_category_names == "Home Area")[0][0])

                            else:
                                print("Color not accounted for rgb({}, {}, {})".format(red, green, blue))

                    self.shelter_matrix = shelter_factor
                    self.shelter_matrix[self.shelter_matrix == 0] = 0.001 # to avoid division by 0 errors
                    #np.save(shelter_cat_url, self.shelter_category)
                    #np.save(shelterMatrixUrl, self.shelter_matrix)
                    #np.save(isWater_url, self.isWater)

            elif self.city == "Paris":
                # changedensmap = Image.open(self.url_modifyPopDens)
                # changedensmap = changedensmap.convert("RGB")

                # 170, 218, 255 = blue = water
                # 199, 233, 199 = green = parks
                # 255, 234, 160 = large roads = yellow
                # 255, 255, 255 = white = roads
                # 232, 232, 232 = grey = residential
                # 254, 247, 224 = light yellow (office districts)
                # 252, 232, 230 = pink, hospital

                self.shelter_category_names = np.array(
                    ["Error", "Office", "Water", "Street", "Event Location", "Home Area", "Small Street", "Meadow"])
                shelter_cat_url = "env/database/paris/shelter_category_waterSetting=" + str(self.waterSetting) + ".npy"
                isWater_url = "env/database/paris/is_water_waterSetting=" + str(self.waterSetting) + ".npy"
                shelterMatrixUrl = "env/database/paris/shelter_matrix_waterSetting=" + str(self.waterSetting) + ".npy"
                try:
                    if self.purge_cache:
                        raise FileNotFoundError
                    #print("loading from db")
                    self.shelter_category = np.load(shelter_cat_url)
                    self.isWater = np.load(isWater_url)  # This is redundant!
                    self.shelter_matrix = np.load(shelterMatrixUrl)
                except FileNotFoundError:
                    for i in range(0, width):
                        for j in range(0, height):
                            coordinates = i, j  # creating tuple as input of the next line
                            red, green, blue = picca.getpixel(
                                coordinates)  # obtaining the values of RGB for every pixel

                            # red_cd, green_cd, blue_cd = changedensmap.getpixel(coordinates)

                            if red == 170 and green == 218 and blue == 255:  # blue, water
                                shelter_factor[i, j] = 0
                                self.isWater[i, j] = 1
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Water")[0][0])
                            elif red == 199 and green == 233 and blue == 199: # green, parks
                                shelter_factor[i, j] = 0
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Meadow")[0][0])
                            elif red == 255 and green == 234 and blue == 160:  # yellow
                                shelter_factor[i, j] = 3.5
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Street")[0][0])
                            elif red == 255 and green == 255 and blue == 255:  # white, small street
                                shelter_factor[i, j] = 2.5
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Small Street")[0][0])

                            elif (red == 253 and green == 246 and blue == 224) or (red == 252 and green == 232 and blue == 230):
                                # pink = hospital = considered office
                                # light yellow = offices
                                shelter_factor[i, j] = 7.5  # offices
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Office")[0][0])

                            elif (red == 232 and green == 232 and blue == 232):  # home / office area
                                shelter_factor[i, j] = 5
                                self.shelter_category[i, j] = int(
                                    np.where(self.shelter_category_names == "Home Area")[0][0])

                            else:
                                print("Color not accounted for rgb({}, {}, {})".format(red, green, blue))

                    self.shelter_matrix = shelter_factor
                    self.shelter_matrix[self.shelter_matrix == 0] = 0.001 # to avoid division by 0 errors
                    #np.save(shelter_cat_url, self.shelter_category)
                    #np.save(shelterMatrixUrl, self.shelter_matrix)
                    #np.save(isWater_url, self.isWater)

            else:
                raise Exception("Sheltering data is not defined for this city")



        # if self.scaling_method == "Normal":
        #     self.Shelter_scaled = self.shelter_matrix[0:self.shelter_matrix.shape[0]:self.scaling_factor,
        #                           0:self.shelter_matrix.shape[1]:self.scaling_factor]
        #
        # elif self.scaling_method == "Real_Average":
        #     width_scaled = math.ceil(self.shelter_matrix.shape[0] / self.scaling_factor)
        #     height_scaled = math.ceil
        #     self.Shelter_scaled = input()

    def removePopDensityFromWater(self):
        '''
        This method removes population density from water.
        '''
        # print("removing pop density from water")
        if self.city == "Delft" or self.city == "NewYork" or self.city == "Paris":
            self.density_matrix[self.isWater == 1] = 0.0
            # self.shelter_matrix[self.isWater == 1] = 2.5
            if len(np.where(self.isWater)[0]) == 0:
                raise Exception("Trying to remove water, but there is no water")
            # print("Not removing water because city is {}".format(self.city))
            # raise Exception("Removing water is only available for Delft")
        else:
            print("Warning, function removePopDensityFromWater called but is not working")

    def modifyPopDensityWater(self):
        '''
        Method that is ran whenever the loadEnvironment class is initialized. It modifies the population density on water.
        '''
        if self.city == "Delft" or self.city == "NewYork" or self.city == "Paris":
            # print("Modifying water")
            self.density_matrix[self.isWater == 1] = 0.01
            # self.shelter_matrix[self.isWater == 1] = 2.5
            if len(np.where(self.isWater)[0]) == 0:
                raise Exception("Trying to remove water, but there is no water")
        else:
            print("Warning, function modifyPopDensityWater called but is not working")



    def statisticsPopulationDensity(self, verbose=True):
        '''
        Outputs statistics of the distribution of people in a city.
        verbose: boolean, whether or not to print the statistics
        '''
        # if self.url_modifyPopDens == None:
        #     raise Exception("Cannot modify population density - no image given")
        # else:
        #     figure = Image.open(self.url_modifyPopDens)
        #     figure = figure.convert('RGB')

        width, height = self.density_matrix.shape  # obtaining dimensions of figure


        # CODE: 0 = home area, 1 = office area, 2 = transport area, 3 = other
        self.area_code = np.zeros((width, height))

        self.population_per_category = np.zeros((len(self.shelter_category_names)))

        self.area = self.grid_size_m_x * self.grid_size_m_y


        width_m = self.width_meters / self.density_matrix.shape[0]
        height_m = self.height_meters / self.density_matrix.shape[1]
        area_m = width_m * height_m

        total_pop = 0
        for i in range(0, width):
            for j in range(0, height):
                total_pop += area_m * self.density_matrix[i, j]

        if verbose:
            print("Total population of {} million people".format(total_pop/1e6))

        for i in range(len(self.area_per_shelter_category)):
            self.area_per_shelter_category[i] = np.count_nonzero(self.shelter_category == i)

        for i in range(0, width):
            for j in range(0, height):
                try:
                    self.population_per_category[self.shelter_category[i, j]] += self.density_matrix[i, j]
                except IndexError:
                    print("debug here")
        if abs(1 - sum(self.population_per_category) / sum(sum(self.density_matrix))) > 0.0001:
            raise Exception("Not all population accounted for in the statistics.")

        if verbose:
            total_pop = sum(self.population_per_category)
            for idx, category in enumerate(self.shelter_category_names):
                relative_dens = self.population_per_category[idx] / self.area_per_shelter_category[idx]
                print("Population in category {} is {} (is {}% of total, relative density = {})".format(category, self.population_per_category[idx], 100*self.population_per_category[idx]/total_pop, relative_dens))


    def generateCityGrid(self, block_width, block_height, street_width, n_blocks_width, n_blocks_height, position_river=None, width_river=5, n_block_river=3):
        '''
        Method that can create a dummy city.
        '''

        self.shelter_category_names = np.array(
            ["Error", "Office", "Water", "Street", "Event Location", "Home Area", "Small Street", "Meadow"])

        total_width = n_blocks_width * block_width + (n_blocks_width - 1) * street_width
        total_height = n_blocks_height * block_height + (n_blocks_height - 1) * street_width

        self.shelter_matrix = np.zeros((total_width, total_height))
        self.density_matrix = np.zeros_like(self.shelter_matrix)

        self.isWater = np.zeros((total_width, total_height))
        self.shelter_category = np.zeros((total_width, total_height), dtype=int)

        type_block = np.zeros((n_blocks_width, n_blocks_height))

        # This can be different in the future!
        type_block[1, 2] = type_block[3, 2] = 1 # meadows

        type_block[1, 1] = type_block[3, 1] = type_block[1, 3] = type_block[3, 3] = type_block[2, 1] = type_block[
            2, 3] = 2 #offices

        for i in range(0, n_blocks_width):
            start_x = 0 + i * (block_width + street_width)
            end_x = block_width + i * (block_width + street_width)
            for j in range(0, n_blocks_height):
                start_y = 0 + j * (block_height + street_width)
                end_y = block_height + j * (block_height + street_width)

                if type_block[i, j] == 1:
                    # category meadow
                    self.density_matrix[start_x:end_x, start_y:end_y] = 2
                    self.shelter_matrix[start_x:end_x, start_y:end_y] = 0.001
                    self.shelter_category[start_x:end_x, start_y:end_y] = int(np.where(self.shelter_category_names == "Meadow")[0][0])
                elif type_block[i, j] == 2:
                    # category office
                    self.density_matrix[start_x:end_x, start_y:end_y] = 5
                    self.shelter_matrix[start_x:end_x, start_y:end_y] = 5
                    self.shelter_category[start_x:end_x, start_y:end_y] = int(np.where(self.shelter_category_names == "Office")[0][0])
                elif type_block[i, j] == 0:
                    # category home
                    self.density_matrix[start_x:end_x, start_y:end_y] = 10
                    self.shelter_matrix[start_x:end_x, start_y:end_y] = 3.5
                    self.shelter_category[start_x:end_x, start_y:end_y] = int(np.where(self.shelter_category_names == "Home Area")[0][0])

        self.shelter_matrix[self.shelter_category == 0] = 2 # TODO make this a better value
        self.density_matrix[self.shelter_category == 0] = 1 # TODO make this a better value
        self.shelter_category[self.shelter_category == 0] = int(np.where(self.shelter_category_names == "Street")[0][0])

        if width_river > 0:
            river = np.zeros((width_river, total_height))
            river_sheltering = np.ones_like(river) * 1e-6
            river_category = np.ones_like(river) * int(np.where(self.shelter_category_names == "Water")[0][0])
            river_category = river_category.astype(int)
            start = int(0 + n_block_river * (block_width + street_width) - street_width / 2)

            if position_river == "left":
                self.shelter_matrix = np.vstack((river_sheltering, self.shelter_matrix))
                self.density_matrix = np.vstack((river, self.density_matrix))
                self.shelter_category = np.vstack((river_category, self.shelter_category))

            elif position_river == "right":
                self.shelter_matrix = np.vstack((self.shelter_matrix, river_sheltering))
                self.density_matrix = np.vstack((self.density_matrix, river))
                self.shelter_category = np.vstack((self.shelter_category, river_category))

            elif position_river == "middle":
                self.shelter_matrix = np.vstack((self.shelter_matrix[0:start, :], river_sheltering, self.shelter_matrix[start:, :]))
                self.density_matrix = np.vstack((self.density_matrix[0:start, :], river, self.density_matrix[start:, :]))
                self.shelter_category = np.vstack((self.shelter_category[0:start, :], river_category, self.shelter_category[start:, :]))

            else:
                raise Exception("unknown position")



    def modifyPopulationDensity(self, target_array):
        '''
         Takes a target_array as input, which should be of the same length as shelter_category_names.
         It moves x % of the population to that area, so the total array should sum to 1.
        '''



        if len(target_array) != len(self.shelter_category_names):
            raise Exception("Modification array should be of same size as category names array")

        total_population = sum(sum(self.density_matrix))

        self.density_matrix_modified = np.zeros_like(self.density_matrix)


        # The weights-list should be pairwise multiplied with the area-size to find the constant

        pop_dens_constant = total_population / sum(np.multiply(self.area_per_shelter_category, target_array))

        for i in range(0, self.width):
            for j in range(0, self.height):
                self.density_matrix_modified[i,j] = target_array[self.shelter_category[i,j]] * pop_dens_constant

        # print(sum(sum(self.density_matrix_modified)) / total_population)

        new_total_population = sum(sum(self.density_matrix_modified))

        # print("Initial total pop = {}, final total pop = {}".format(total_population, new_total_population))

        diff_pop = abs((new_total_population / total_population) - 1)
        if diff_pop > 0.05:
            raise Exception("Total population count changed more than 5% while moving people around - this can't be right!")

        if diff_pop > 0.001:
            print("WARNING: Total population count changed by {}%".format(diff_pop*100))

        self.density_matrix = self.density_matrix_modified.copy()

    def scaleMap(self, na, scaling_factor, scaling_method="average", file_name=""):
        '''

        @param na: numpy array of the map that is to be scaled
        @param scaling_factor: factor by which this map should be scaled
        @param scaling_method: method to use (mode / average)
        @param file_name: file name in which the map can be stored.
        @return:
        '''
        file_loc = "env/database/scaled_maps/" + "c=" + self.city + "_n=" + file_name + "_s=" + str(scaling_factor) + "_m=" + scaling_method + "_ws=" + str(self.waterSetting) + ".npy"
        try:
            if self.purge_cache:
                raise FileNotFoundError
            #print("loading from db")

            scaled_map = np.load(file_loc)
            #print("loading from db")
        except FileNotFoundError:
            if scaling_factor == 1:
                return na

            block_shape = (scaling_factor, scaling_factor)

            width_scaled = math.floor(na.shape[0] / block_shape[0])
            height_scaled = math.floor(na.shape[1] / block_shape[1])

            width_cropped = width_scaled * block_shape[0]
            height_cropped = height_scaled * block_shape[1]

            view = view_as_blocks(na[:width_cropped, :height_cropped], block_shape)
            flatView = view.reshape(view.shape[0], view.shape[1], -1)

            if scaling_method == "average":
                scaled_map = np.average(flatView, axis=2)
            elif scaling_method == "mode":
                scaled_map = stats.mode(flatView, axis=2)[0].reshape((width_scaled, height_scaled))
            elif scaling_method == "original_AABMS":
                scaled_map = na[0:na.shape[0]:scaling_factor, 0:na.shape[1]:scaling_factor]
            else:
                raise Exception("Unknown scaling mode.")

            #np.save(file_loc, scaled_map)

        return scaled_map


if __name__ == "__main__":
    '''
    Code here is only used for creating plots and evaluating whether the loadEnvironment class is working as it should.
    '''
    import matplotlib.pyplot as plt

    env = loadEnvironment(city="Paris", scaling_factor=1, risk_map_resolution=1, water_setting=0)

    beta = 34
    alpha = 32000
    width = env.shelter_matrix.shape[0]
    height = env.shelter_matrix.shape[1]
    value_table = np.zeros((width,height))
    '''Calculate the risk at each location'''
    for i in range(width):
        for j in range(height):
            ps = env.shelter_matrix[i, j]
            # Impact kinetic energy
            v_terminal = 50
            Eimp = 0.5 * 3 * v_terminal ** 2
            # alpha = 32000  # impact energy required for a fatality probability of 50% when ps = 6
            # beta = 34  # impact energy needed to cause a fatality when ps approaches zero

            # Falality probability
            k = min(1, pow((beta / Eimp), 3 / ps))
            value = pow((alpha / beta), 0.5) * pow((beta / Eimp), 3 / (ps))
            pf = (1 - k) / (1 - 2 * k + value)
            value = pf * env.density_matrix[i, j]  # density_matrix unit = people / m^2
            # The unit of value = the amount of people that would die, given an impact in this place
            value_table[i, j] = value

    '''Plot the shelter factors at each location'''
    plt.imshow(env.shelter_matrix.transpose())
    plt.colorbar()
    plt.savefig("shelterParis.png")
    plt.show()

    '''Plot the pop density at each location'''
    plt.imshow(env.density_matrix.transpose())
    plt.colorbar()
    plt.savefig("densityParis.png")
    plt.show()

    '''Plot the shelter factors at each location'''
    plt.imshow(value_table.transpose(), cmap="Reds")
    plt.colorbar()
    plt.savefig("riskParis.png")
    plt.show()


    env.statisticsPopulationDensity(verbose=True)
