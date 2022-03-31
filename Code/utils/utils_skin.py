class e_skin_Sensor:
    def __init__(self, sensornumber, rectangle_width, color, position, transparency=1):
        self.sensornumber = sensornumber
        self.rectangle_side = rectangle_width
        self.color = color
        self.transparency = transparency
        self.position = position

    def draw_my_sensor(self, img):
        center = self.position
        overlay = img.copy()
        pts = np.array([[center[0] - self.rectangle_side * 1 / 2, center[1] - self.rectangle_side * 1 / 2],
                        [center[0] + self.rectangle_side * 1 / 2, center[1] - self.rectangle_side * 1 / 2],
                        [center[0] + self.rectangle_side * 1 / 2, center[1] + self.rectangle_side * 1 / 2],
                        [center[0] - self.rectangle_side * 1 / 2, center[1] + self.rectangle_side * 1 / 2]], np.int32)
        if isinstance(self.color,str):
            self.color = self.hex_to_rgb(self.color)
        cv.fillPoly(overlay, [pts], color=self.color)
        cv.addWeighted(overlay, self.transparency, img, 1 - self.transparency,
                       0, img)
    def hex_to_rgb(self,value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    def change_my_color(self, color, img, transparency=1):
        self.color = color
        self.transparency = transparency
        self.draw_my_sensor(img)
    def give_position(self):
        return self.position

class Marker:
    def __init__(self, id, position, std, radius, img, color=(255, 255, 255), transparency=1):
        self.id = id
        self.position = position
        self.radius = radius
        self.color = color
        self.transparency = transparency
        overlay = img.copy()
        cv.circle(overlay, center=self.position, radius=self.radius, color=self.color, thickness=-1)
        start_point_x = (position[0] + std[0], position[1])
        end_point_x = (position[0] - std[0], position[1])
        start_point_y = (position[0], position[1] + std[1])
        end_point_y = (position[0], position[1] - std[1])
        cv.line(overlay, start_point_x, end_point_x, color=self.color, thickness=3)
        cv.line(overlay, start_point_y, end_point_y, color=self.color, thickness=3)
        cv.addWeighted(overlay, self.transparency, img, 1 - self.transparency,
                       0, img)


class Triangle:
    def __init__(self, sensors, upper_corner, side_length, internal_angle, rectangle_width, color_here, mirror=False):
        self.upper_corner_x = upper_corner[0]
        self.upper_corner_y = upper_corner[1]
        self.side_length = side_length
        self.internal_angle = internal_angle
        self.sensor_collection = []
        self.mirror = mirror
        division = 20
        slices = 5
        initial_shiftx = self.upper_corner_x
        initial_shifty = self.upper_corner_y
        shifty = 20

        for sensor_ix, sensor in enumerate(sensors):
            if sensor_ix < 4:
                row = 0
                sensor_ix_shift = 0
            elif sensor_ix < 7:
                row = 1
                sensor_ix_shift = 4
            elif sensor_ix < 9:
                row = 2
                sensor_ix_shift = 7
            elif sensor_ix < 12:
                row = 3
                sensor_ix_shift = 9
            else:
                print('Sensors are more than expected')

            slices = 5 - row
            shiftx = self.side_length / 5 * 0.833 - 25
            shifty = 40 + row * 40
            # for slice in range(slices)[:-1]:
            position = [shiftx + initial_shiftx + self.side_length / 5 * (sensor_ix - sensor_ix_shift) * 0.833 * (
                        int(mirror == False) - int(mirror == True)) + division,
                        shifty + initial_shifty + self.side_length / 5 * (sensor_ix - sensor_ix_shift) * 0.5]
            self.sensor_collection.append(e_skin_Sensor(sensor, rectangle_width, color_here, position))

    def draw_my_triangle(self, img):
        for sensor in self.sensor_collection:
            sensor.draw_my_sensor(img)

    def change_color(self, sensor, color, img, transparency=1):
        self.sensor_collection[sensor].change_my_color(color, img, transparency=transparency)

    def give_position_vector(self):
        position_vector = np.zeros([len(self.sensor_collection), 2])
        for sens_ix, sensor in enumerate(self.sensor_collection):
            position_vector[sens_ix, :] = sensor.give_position()
        return position_vector


class Skin:
    def __init__(self, triangle_side, rectangle_side):
        # print('Creating the skin')
        self.triangle_side = triangle_side
        self.rectangle_side = rectangle_side
        self.x_center = None
        self.y_center = None
        self.dimensions = None
        self.sensors_array = np.array(
            [[121, 123, 112, 113, 120, 115, 114, 119, 116, 117], [49, 50, 52, 53, 48, 51, 55, 59, 56, 57],
             [97, 96, 107, 105, 98, 99, 104, 100, 103, 101], [69, 68, 66, 65, 71, 67, 64, 72, 75, 73],
             [145, 146, 148, 149, 144, 147, 151, 155, 152, 153], [81, 82, 84, 85, 80, 83, 87, 91, 88, 89],
             [129, 128, 139, 137, 130, 131, 136, 132, 135, 133], [165, 166, 162, 161, 167, 163, 160, 168, 171, 169],
             [217, 219, 208, 209, 216, 211, 210, 215, 212, 213], [181, 183, 184, 185, 180, 179, 187, 178, 176, 177],
             [193, 192, 203, 201, 194, 195, 200, 196, 199, 197], [229, 228, 226, 225, 231, 227, 224, 232, 235, 233],
             [25, 27, 16, 17, 24, 19, 18, 23, 20, 21], [245, 247, 248, 249, 244, 243, 251, 242, 240, 241],
             [41, 40, 39, 37, 43, 35, 36, 32, 34, 33], [9, 8, 7, 5, 11, 3, 4, 0, 2, 1]])
        self.markers_array = []
        self.addresses = np.array([
            [17301751, 17301749, 17301747, 17301745, 17301743, 17301741, 17301739, 17301737, 17301735, 17301733,
             17301731, 17301729],
            [17301719, 17301717, 17301715, 17301713, 17301711, 17301709, 17301707, 17301705, 17301703, 17301701,
             17301699, 17301697],
            [17301623, 17301621, 17301619, 17301617, 17301615, 17301613, 17301611, 17301609, 17301607, 17301605,
             17301603, 17301601],
            [17301655, 17301653, 17301651, 17301649, 17301647, 17301645, 17301643, 17301641, 17301639, 17301637,
             17301635, 17301633],
            [17301783, 17301781, 17301779, 17301777, 17301775, 17301773, 17301771, 17301769, 17301767, 17301765,
             17301763, 17301761],
            [17301815, 17301813, 17301811, 17301809, 17301807, 17301805, 17301803, 17301801, 17301799, 17301797,
             17301795, 17301793],
            [17301847, 17301845, 17301843, 17301841, 17301839, 17301837, 17301835, 17301833, 17301831, 17301829,
             17301827,
             17301825],
            [17301687, 17301685, 17301683, 17301681, 17301679, 17301677, 17301675, 17301673, 17301671, 17301669,
             17301667,
             17301665],
            [17301943, 17301941, 17301939, 17301937, 17301935, 17301933, 17301931, 17301929, 17301927, 17301925,
             17301923,
             17301921],
            [17301911, 17301909, 17301907, 17301905, 17301903, 17301901, 17301899, 17301897, 17301895, 17301893,
             17301891,
             17301889],
            [17301879, 17301877, 17301875, 17301873, 17301871, 17301869, 17301867, 17301865, 17301863, 17301861,
             17301859,
             17301857],
            [17301975, 17301973, 17301971, 17301969, 17301967, 17301965, 17301963, 17301961, 17301959, 17301957,
             17301955,
             17301953],
            [17301591, 17301589, 17301587, 17301585, 17301583, 17301581, 17301579, 17301577, 17301575, 17301573,
             17301571,
             17301569],
            [17301559, 17301557, 17301555, 17301553, 17301551, 17301549, 17301547, 17301545, 17301543, 17301541,
             17301539,
             17301537],
            [17301527, 17301525, 17301523, 17301521, 17301519, 17301517, 17301515, 17301513, 17301511, 17301509,
             17301507,
             17301505],
            [17302007, 17302005, 17302003, 17302001, 17301999, 17301997, 17301995, 17301993, 17301991, 17301989,
             17301987,
             17301985]]).flatten()
        self.addresses = self.Convert_addresses()
        self.triangle_collection = []
        self.position_matrix = []
        self.lateral_inh_matrix = []
        for triangle in range(len(self.sensors_array)):
            # for triangle in range(1):
            if triangle == 0:
                self.triangle_collection.append(Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(0, 0),
                                                         side_length=self.triangle_side, internal_angle=30,
                                                         rectangle_width=self.rectangle_side,
                                                         color_here=(255, 255, 255)))
            elif triangle == 1:
                self.triangle_collection.append(Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(0, 220),
                                                         side_length=self.triangle_side, internal_angle=30,
                                                         rectangle_width=self.rectangle_side,
                                                         color_here=(255, 255, 255)))
            elif triangle == 2:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(130, 110),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 3:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(130, 330),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 4:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(190, 110),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255)))
            elif triangle == 5:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(190, 330),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255)))
            elif triangle == 6:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(315, -5),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 7:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(315, 220),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 8:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(375, -5),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=False))
            elif triangle == 9:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(375, 220),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=False))
            elif triangle == 11:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(500, 330),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 12:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(565, 110),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=False))
            elif triangle == 10:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(500, 105),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            elif triangle == 13:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(565, 330),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=False))
            elif triangle == 14:
                self.triangle_collection.append(Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(695, 0),
                                                         side_length=self.triangle_side, internal_angle=30,
                                                         rectangle_width=self.rectangle_side,
                                                         color_here=(255, 255, 255), mirror=True))
            elif triangle == 15:
                self.triangle_collection.append(
                    Triangle(sensors=self.sensors_array[triangle, :], upper_corner=(695, 225),
                             side_length=self.triangle_side, internal_angle=30,
                             rectangle_width=self.rectangle_side,
                             color_here=(255, 255, 255), mirror=True))
            else:
                print('Something is wrong with the triangles')
                print('Triangle expected:' + str(triangle))

    def draw_my_skin(self, img):
        for triangle in self.triangle_collection:
            triangle.draw_my_triangle(img)
    def Get_Dimensions(self):
        if self.dimensions == None:
            self.dimensions = ()
            _ = self.give_position_matrix()
            self.dimensions[0] = self.position_matrix[:,:,0].max() - self.position_matrix[:,:,0].min()
            self.dimensions[1] = self.position_matrix[:,:,1].max() - self.position_matrix[:,:,1].min()
        return self.dimensions

    def draw_marker(self, position, std, img, radius=1, color=(255, 255, 255), transparency=1):
        self.markers_array.append(
            Marker(id=len(self.markers_array), position=position, std=std, radius=radius, color=color,
                   transparency=transparency, img=img))


    def reset_colors(self, img, color=(53, 119, 48), transparency=1):
        for sensor in self.sensors_array:
            self.change_color(sensor, color, img, transparency=transparency)

    def Find_Center_of_Skin(self, return_enable=False):
        if self.x_center == None:
            matrix = self.give_position_matrix()
            x_extreme_low = matrix[:, :, 0].min()
            x_extreme_high = matrix[:, :, 0].max()
            self.x_center = (x_extreme_high + x_extreme_low) / 2
        if self.y_center == None:
            y_extreme_low = matrix[:, :, 1].min()
            y_extreme_high = matrix[:, :, 1].max()
            self.y_center = (y_extreme_high + y_extreme_low) / 2
        if return_enable == True:
            return [self.x_center, self.y_center]
    def mm_to_pixels(self,length):
        if (isinstance(length, list)) | (isinstance(length,np.ndarray)):
            output = []
            for element in length:
                output.append(element*8)
            return output
        else:
            return length*8
    def pixels_to_mm(self,length):
        if (isinstance(length, list)) | (isinstance(length,ndarray)):
            output = []
            for element in length:
                output.append(element / 8)
            return output
        else:
            return length / 8
    def Find_Distance_from_center(self, sensors):
        self.Find_Center_of_Skin()
        sensors = self.Check_Position_Consinstency(sensors=sensors)
        my_array = np.zeros([len(sensors), 2])
        for sens_ix, sensor in enumerate(sensors):
            triangle_n, sensor_n = self.find_sensor(sensor=sensor)
            triangle_n = triangle_n[0]
            sensor_n = sensor_n[0]
            my_array[sens_ix, 0] = self.position_matrix[triangle_n, sensor_n, 0] - self.x_center
            my_array[sens_ix, 1] = self.position_matrix[triangle_n, sensor_n, 1] - self.y_center
        return my_array

    def Check_Position_Consinstency(self, sensors):
        sensors_cleaned = []
        for sensor in sensors:
            triangle_n, sensor_n = self.find_sensor(sensor=sensor)
            if len(triangle_n) > 0:
                sensors_cleaned.append(sensor)
        return sensors_cleaned

    def Statistics_on_Skin(self, sensors):
        distances = self.Find_Distance_from_center(sensors=sensors)
        mean_x = np.mean(distances[:, 0])
        std_x = np.std(distances[:, 0])
        mean_y = np.mean(distances[:, 1])
        std_y = np.std(distances[:, 1])
        dict = {'mean_x': mean_x, 'std_x': std_x, 'mean_y': mean_y, 'std_y': std_y}
        return dict

    def Triangle_Index(self, sensors):
        triangle_indexes = np.zeros([len(self.triangle_collection)])
        for sensor in sensors:
            triangle, sensor_here = self.find_sensor(sensor=sensor)
            if len(triangle) > 0:
                triangle_indexes[triangle[0]] += 1
        return triangle_indexes / 10

    def Clustering_Indexes(self, sensors):
        # distances = self.Find_Distance_from_center(sensors=sensors)
        self.give_neighbour(distance=200)
        # print(self.lateral_inh_matrix)
        cluster_index = 0
        counter = 0
        for sensor_ix0 in sensors:
            triangle0, sensor0 = self.find_sensor(sensor=sensor_ix0)
            if len(triangle0) > 0:
                for sensor_ix1 in sensors:
                    triangle1, sensor1 = self.find_sensor(sensor=sensor_ix1)
                    if len(triangle1) > 0:
                        cluster_index = cluster_index + self.lateral_inh_matrix[
                            triangle0[0] * self.position_matrix.shape[1] + sensor0[0], triangle1[0] *
                            self.position_matrix.shape[1] + sensor1[0]]
                        counter += 1
        return counter / cluster_index

    def Convert_addresses(self):
        and_value = 0xff
        masked_addresses = np.bitwise_and(np.right_shift(self.addresses, 1), and_value)
        return np.reshape(masked_addresses, masked_addresses.size)

    def Position_to_Number(self, Position):
        taxel_active = []
        if isinstance(Position, str):
            Position = list(map(int, Position))
        try:
            len(Position)
            for index, taxel in enumerate(Position):
                taxel_active.append(self.addresses[int(taxel)])
        except TypeError:
            taxel_active = self.addresses[int(Position)]
        return taxel_active

    def Number_to_Position(self, Number):
        taxel_active = []
        if isinstance(Number,int):
            positions = np.where(self.addresses == Number)[0]
            if len(positions) == 0:
                print("The taxel you are searching (" + str(Number) + ") doesn't exist")
                raise ValueError
            else:
                taxel_active.append(positions[0])
        elif isinstance(Number,np.float64):
            positions = np.where(self.addresses == int(Number))[0]
            if len(positions) == 0:
                print("The taxel you are searching (" + str(Number) + ") doesn't exist")
                raise ValueError
            else:
                taxel_active.append(positions[0])
        elif isinstance(Number,list) | isinstance(Number,ndarray):
            for taxel in Number:
                # print(taxel)
                positions = np.where(self.addresses == taxel)[0]
                if len(positions) == 0:
                    print("The taxel you are searching (" + str(taxel)  + ") doesn't exist")
                    raise ValueError
                else:
                    taxel_active.append(positions[0])
        else:
            print('Input values should be a number of a list.')
            raise ValueError
        return taxel_active

    def find_sensor(self, sensor):
        return np.where(self.sensors_array == sensor)

    def change_color(self, sensor, color, img, transparency=1):
        which_triangle, which_sensor = self.find_sensor(sensor)
        try:

            self.triangle_collection[which_triangle[0]].change_color(sensor=which_sensor[0], color=color, img=img,
                                                                     transparency=transparency)
        except IndexError:
            if 1 == 0:
                print('The index ' + str(sensor) + ' has not been found, probably is not a pressure sensor')

    def give_position_matrix(self, sensors=[]):
        if len(self.position_matrix) == 0:
            self.position_matrix = np.zeros([self.sensors_array.shape[0], self.sensors_array.shape[1], 2])
            for tri_ix, triangle in enumerate(self.triangle_collection):
                self.position_matrix[tri_ix, :, :] = triangle.give_position_vector()
        if len(sensors) > 0:
            collection = np.zeros([len(sensors), 3])
            for sens_ix, sensor in enumerate(sensors):
                which_triangle, which_sensor = self.find_sensor(sensor)
                if len(which_triangle) > 0:
                    collection[sens_ix, 0:2] = self.position_matrix[which_triangle[0], which_sensor[0], :]
                    collection[sens_ix, 2] = int(sensor)
                else:
                    collection[sens_ix, 0] = np.NAN
                    collection[sens_ix, 1] = np.NAN
                    collection[sens_ix, 2] = int(sensor)
            return collection
        else:
            return self.position_matrix

    def polar_indexes(self, sensors):
        difference = self.calculate_polar_distances(sensors=sensors)
        avg_angle = difference[:, :, 1].nanmean()
        std_angle = difference[:, :, 1].nanstd()
        avg_mean = difference[:, :, 0].nanmean()
        std_mean = difference[:, :, 0].nanstd()
        dict = {'avg_angle': avg_angle, 'std_angle': std_angle, 'avg_mean': avg_mean, 'std_mean': std_mean}
        return dict

    def calculate_polar_distances(self, sensors):
        positions = self.give_position_matrix(sensors=sensors)
        positions = positions[~np.isnan(positions).any(axis=1)]
        difference = np.zeros([positions.shape[0], positions.shape[0], 2])
        difference[:, :, :] = np.NAN
        sorted_ix = np.argsort(positions[:, 0])
        positions_sorted = positions[sorted_ix, :]
        for sens_ix in range(positions_sorted.shape[0]):
            remaining = np.where(positions_sorted[:, 0] > positions_sorted[sens_ix, 0])
            print(remaining)
            for sens_ix2 in range(positions_sorted[remaining].shape[0]):
                dx = positions_sorted[sens_ix, 0] - positions_sorted[sens_ix2, 0]
                dy = positions_sorted[sens_ix, 1] - positions_sorted[sens_ix2, 1]
                difference[sens_ix, sens_ix2, 0] = np.sqrt(dx ** 2 + dy ** 2)
                difference[sens_ix, sens_ix2, 1] = np.rad2deg(np.arctan(dy / dx))
        return difference

    def give_neighbour(self, distance=50, sensors=[]):
        if len(self.lateral_inh_matrix) == 0:
            if len(self.position_matrix) == 0:
                self.position_matrix = self.give_position_matrix()
            self.lateral_inh_matrix = np.zeros(
                [self.position_matrix.shape[0] * self.position_matrix.shape[1],
                 self.position_matrix.shape[0] * self.position_matrix.shape[1]])
            for row1 in range(self.position_matrix.shape[0]):
                for column1 in range(self.position_matrix.shape[1]):
                    for row2 in range(self.position_matrix.shape[0]):
                        for column2 in range(self.position_matrix.shape[1]):
                            d = np.sqrt((self.position_matrix[row1, column1, 0] - self.position_matrix[
                                row2, column2, 0]) ** 2 + (
                                                self.position_matrix[row1, column1, 1] - self.position_matrix[
                                            row2, column2, 1]) ** 2)
                            if isinstance(distance, tuple) | isinstance(distance,list):
                                if (abs(d) > abs(distance[0])) & (abs(d) < abs(distance[1])):
                                    self.lateral_inh_matrix[
                                        row1 * self.position_matrix.shape[1] + (column1), row2 *
                                        self.position_matrix.shape[
                                            1] + column2] = d
                            else:
                                if abs(d) < abs(distance):
                                    self.lateral_inh_matrix[
                                        row1 * self.position_matrix.shape[1] + (column1), row2 *
                                        self.position_matrix.shape[
                                            1] + column2] = d
        if isinstance(sensors, int):
            sensors = [sensors]
        elif isinstance(sensors, np.int64):
            sensors = [int(sensors)]
        if len(sensors) > 0:
            collection = []
            for sensor in sensors:
                which_triangle, which_sensor = self.find_sensor(sensor)
                which_neighbours = np.where(
                    self.lateral_inh_matrix[which_triangle[0] * self.position_matrix.shape[1] + which_sensor[0], :] > 0)
                for neighbour in which_neighbours[0]:
                    triangle = int(np.floor(neighbour / self.position_matrix.shape[1]))
                    sensor_number = neighbour - triangle * self.position_matrix.shape[1]
                    sensor_id = self.sensors_array[triangle, sensor_number]
                    collection.append(sensor_id)
            return collection

        else:
            return self.lateral_inh_matrix








def Plot_Skin_Stimuli(my_skin, Image_PATH, Image_FILE, analog_norm, orientation_array, analog_clock,
                      bar = None, texture = None, loop = False):
    """
    This function plots on the image of the e-skin the excited sensors during the onset of the stimulus.
    The input can be recorded from the real skin or generated in code. In the latter case the script provides the
    possibility to visualize the simulated stimuli. In this case feed to the function the stimulus properties through
    the variables bar (if a bar is the stimulus) or texture(if texture is the stimulus).


    :param my_skin: The object containing all the information about how the skin is made.
    :param Image_PATH: (string) Where to find the image of the skin patch.
    :param Image_FILE: (string) The name of the skin patch's image file.
    :param analog_norm: (array) The array [sensors x time] containing which sensors is excited for each time instant.
    :param orientation_array: (list) A list [[time,stimulus_id],[time,stimulus_id]] that contains the times of each stimulus.
    :param analog_clock: (brian time variable) The time instant step (if the analog_norm is recorded is the sampling period).
    :param bar: (list) The collection of the coordinates of the rectangles applied on the skin for each stimulus.
    :param texture: The collection of the coordinates of the multiple rectangles (composing the texture) applied on the skin for each stimulus.
    :return: The function doesn't return anything
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    # print(orientation_array)
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    img = cv.imread(Image_PATH + Image_FILE)
    img2 = cv.imread(Image_PATH + Image_FILE)
    my_skin.draw_my_skin(img)
    cv.imshow('Touch', img)
    cv.startWindowThread()
    sensors = my_skin.Position_to_Number([i for i in range(192)])
    in_the_loop = True
    while(in_the_loop == True):
        stimulus = 0
        for i in range(0, len(analog_norm.T)):
            orientation_position = orientation_array[stimulus][1]
            if stimulus < len(orientation_array)-1:
                if i * analog_clock +1*analog_clock > orientation_array[stimulus][0]:
                    stimulus += 1
                    # print(stimulus)
                    cv.imshow('Touch', img)
            testo = 'T:' + str(i * analog_clock) + '   Orientation: ' + str(orientation_position) + ' degree'
            img = np.copy(img2)

            my_skin.reset_colors(img=img)
            # print(sensors)
            if bar != None:
                hh = [bar[stimulus][0][0],bar[stimulus][0][1]]
                lh = [bar[stimulus][1][0], bar[stimulus][1][1]]
                ll = [bar[stimulus][2][0], bar[stimulus][2][1]]
                hl = [bar[stimulus][3][0], bar[stimulus][3][1]]
                pts = np.array([hh,lh,ll,hl], np.int32)
                # pts = pts.reshape((-1, 1, 2))
                cv.fillPoly(img,[pts],color = (0,255,0))
            elif texture != None:
                geometry = texture[stimulus]
                for element in geometry:
                    hh = [element[0][0], element[0][1]]
                    lh = [element[1][0], element[1][1]]
                    ll = [element[2][0], element[2][1]]
                    hl = [element[3][0], element[3][1]]
                    pts = np.array([hh, lh, ll, hl], np.int32)
                    # pts = pts.reshape((-1, 1, 2))
                    cv.fillPoly(img, [pts], color=(0, 255, 0))
            for sens_ix, sensor in enumerate(sensors):
                my_skin.change_color(sensor=sensor, color=(0, 0, 255), img=img, transparency=analog_norm[sens_ix, i])
            cv.putText(img, testo,
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       lineType)
            cv.imshow('Touch', img)
            cv.waitKey(1)
        if loop == False:
            in_the_loop = False

def Plot_Skin_RF(my_skin, Image_PATH, Image_FILE, weights, layer0, layer1, statistic_collection = [], show_here= False, multiple_plots = True, save = [], my_colors = []):
    '''
    The function plots on the image of the e-skin the receptive fields (RF), coded by color. Each color is a receptive field.
    :param my_skin: The object containing all the information about how the skin is made.
    :param Image_PATH: (string) Where to find the image of the skin patch.
    :param Image_FILE: (string) The name of the skin patch's image file.
    :param weights: (array) The RF connections expressed in a [sensors x RF] matrix. The values indicate the strength of the connection.
    :param layer0: (int) The number of sensors on the skin (also the columns of weights).
    :param layer1: (int) The number of RF (also the rows of weights).
    :param show_here: (boolean) Define if figure should be plot during the script execution.
    :param multiple_plots: (boolean) Define if the RF should be plot on one skin image or multiple (one for RF)
    :param save: (str) If defined, saves the plot(s) in the explicited folder
    :return: It returns the matrix of the image
    '''
    image_rf = []
    cv.startWindowThread()

    # my_colors = {'violet': (108, 47, 108), 'blue': (22.7, 25.9, 44.7), 'ocra': (167, 107, 73), 'red': (168, 47, 63),
    #
    if (len(my_colors) < weights.shape[1]):
        import matplotlib._color_data as mcd
        wanted_colors = weights.shape[1]
        my_colors = mcd.CSS4_COLORS

        # my_colors_keys = list(my_colors.keys())
        my_colors_keys = [list((my_colors.keys()))[int(i * int(np.floor(len(my_colors) / wanted_colors)))] for i in range(wanted_colors)]
    else:
        my_colors_keys = list(my_colors.keys())
    if multiple_plots == False:
        image_rf = cv.imread(Image_PATH + Image_FILE)
    for neuron1 in range(layer1):
        # img = cv.imread(Image_PATH + Image_FILE)
        if multiple_plots == True:
            image_rf.append(cv.imread(Image_PATH + Image_FILE))

        # my_skin.draw_my_skin(image_rf[neuron1])
        # my_skin.reset_colors(image_rf[neuron1])
        if (show_here == True) & (multiple_plots == True):
            cv.imshow(str(neuron1), image_rf[neuron1])

        for neuron0 in range(layer0):
            # my_skin.draw_my_skin(image_rf[neuron1])
            if multiple_plots == True:
                image_rf_here = image_rf[neuron1]
            else:
                image_rf_here = image_rf
            sensor = my_skin.Position_to_Number(neuron0)
            if multiple_plots == True:
                my_skin.reset_colors(image_rf_here)
            my_skin.change_color(sensor=sensor, color=my_colors[my_colors_keys[neuron1]], img=image_rf_here,
                                 transparency=weights[neuron0, neuron1]*int(weights[neuron0, neuron1] > 0))
            if (show_here == True) & (multiple_plots == True) :
                cv.imshow(str(neuron1), image_rf_here)
        if len(statistic_collection) > 0:
            statistic_RF = statistic_collection[neuron1]
            position_marker = (int(statistic_RF['mean_x'] + my_skin.x_center),int(statistic_RF['mean_y'] + my_skin.y_center))
            std_marker = (int(statistic_RF['std_x']),int(statistic_RF['std_y']))
            my_skin.draw_marker(position = position_marker, std = std_marker, img = image_rf_here, radius = 10, color = my_colors[my_colors_keys[neuron1]], transparency = 1)
    if show_here == True:
        if multiple_plots == False:
            cv.imshow('RF', image_rf)
        cv.waitKey(0)
    # cv.waitKey(0)
        cv.destroyAllWindows()

    if isinstance(save, str):
        if multiple_plots == True:
            for i,image in enumerate(image_rf):
                cv.imwrite(save + "_" + str(i) + ".png",image)
        else:
            cv.imwrite(save + ".png",image_rf)
    return image_rf


def Plot_neigbours(my_skin,img, sensor, distance=50):
    '''
    The function plots the neighbours of a given sensor on the skin image.
    :param my_skin: The object containing all the information about how the skin is made.
    :param img: The matrix of pixels to modify
    :param sensor: (int) The sensor (id) we want to use as center
    :param distance: (int) The maximum distance the neighbour sensors can be
    :return: The function doesn't return anything
    '''
    my_colors = {'violet': (108, 47, 108), 'blue': (22.7, 25.9, 44.7), 'ocra': (167, 107, 73), 'red': (168, 47, 63),
                 'green': (68, 150, 42), 'orange': (156, 81, 31), 'boh': (110, 150, 20)}
    neighbour_ids = my_skin.give_neighbour(sensors=sensor, distance=distance)
    my_skin.change_color(sensor=sensor, color=my_colors['red'], img=img)
    for sensor in neighbour_ids:
        my_skin.change_color(sensor=sensor, color=my_colors['blue'], img=img)
    cv.imshow('Neighbours', img)
    cv.waitKey(0)

def Create_Stimulus(my_skin,orientation, length, width, shift = [0,0]):
    '''
    The function create a rectangle length x width, shifted of shift from the center and with a given angle from the horizontal position.
    :param my_skin: The object containing all the information about how the skin is made.
    :param orientation: the orientation of the bar expressed in degree.
    :param length: the length of the bar expressed in pixels.
    :param width: the width of the bar expressed in pixels.
    :param shift: the shift from the center of the skin expressed in pixels.
    :return: it returns an array composed of the 4 vertices.
    '''
    length = my_skin.mm_to_pixels(length)
    width = my_skin.mm_to_pixels(width)
    shift = my_skin.mm_to_pixels(shift)
    center = my_skin.Find_Center_of_Skin(return_enable=True)
    orientation = deg2rad(orientation)
    length_side = np.sqrt((length**2/4) + (width**2/4))
    angle_side = np.arctan2(width,length)
    extreme_hh = [shift[0] + center[0] + length_side * np.cos(orientation + angle_side),shift[1] + center[1] +
                  length_side * np.sin(orientation + angle_side)]
    extreme_lh = [shift[0] + center[0] + length_side * np.cos(np.pi + orientation - angle_side),shift[1] + center[1] +
                  length_side * np.sin(np.pi + orientation - angle_side)]
    extreme_ll = [shift[0] + center[0] + length_side * np.cos(np.pi + orientation + angle_side),shift[1] + center[1] +
                  length_side * np.sin(np.pi + orientation + angle_side)]
    extreme_hl = [shift[0] + center[0] + length_side * np.cos(orientation - angle_side),shift[1] + center[1] +
                  length_side * np.sin(orientation - angle_side)]


    return [extreme_hh,extreme_lh,extreme_ll,extreme_hl]

def Generate_Texture(my_skin, active_trials,length, width,space,number,depth = 1, texture_shift = (0,0),
                     orientation = 0, geometry_coordinates = False, moving_texture = False, dataset = 'real', noise = None):
    '''
    The function generate a texture made of multiple (many as number) rectangles length x width, spaced between them of space, rotated of orientation.
    :param my_skin: The object containing all the information about how the skin is made.
    :param active_trials: The collection of real responses of sensors on the skin.
    :param length: The length of each bar in pixels.
    :param width: The width of each bar in pixels.
    :param space: The space between each bar in pixels.
    :param number: The number of bars of which the texture is made.
    :param depth: The pressure level at which the bar is pressed on the skin (not implemented yet)
    :param orientation: The rotation of the texture with respect to the center
    :param geometry_coordinates: If True, the function returns an array with the coordinates of each bar of which the texture is made.
    :return: It return the temporal sequence of one texture applied on the skin. If geometry_coordinates is True it returns also the coordinates of the bars.
    '''
    excited_sensors = np.array([])
    if moving_texture == True:
        dataset = 'simulated'

    if geometry_coordinates == True:
        geometry = []
    for i in range(number):
        shift = [-((space + width/2)*(number/2-i) +texture_shift)*np.sin(-np.deg2rad(orientation) + np.pi) + width/2 ,
                 -((space + width/2)*(number/2-i) + texture_shift)*np.cos(-np.deg2rad(orientation) + np.pi)]
        stimulus = Create_Stimulus(my_skin = my_skin,orientation = orientation, length = length, width = width,shift = shift)
        if len(excited_sensors) == 0:
            excited_sensors = Generate_Response(stimulus = stimulus,
                                                active_dict = active_trials,
                                                my_skin= my_skin,
                                                moving_texture = moving_texture,
                                                dataset = dataset, noise = noise)
        else:
            new_value =  Generate_Response(stimulus = stimulus,
                                                active_dict = active_trials,
                                                my_skin= my_skin,
                                                moving_texture = moving_texture,
                                                dataset = dataset, noise = noise)
            if new_value.shape[1] > excited_sensors.shape[1]:
                new_value = new_value[:,:excited_sensors.shape[1]]
            elif new_value.shape[1] < excited_sensors.shape[1]:
                excited_sensors = excited_sensors[:,:new_value.shape[1]]
            superpositions = np.where((excited_sensors - new_value) < 25692250)
            excited_sensors += new_value
            excited_sensors[superpositions] = new_value[superpositions]
        if geometry_coordinates == True:
            geometry.append(stimulus)
    if geometry_coordinates == True:
        return excited_sensors,geometry
    else:
        return excited_sensors

def Generate_Response(my_skin,stimulus,active_dict,max_len_trial = 680, dataset = 'real', moving_texture = False, noise = None):
    '''

    :param my_skin: The object containing all the information about how the skin is made.
    :param stimulus: The coordinates of the bar applied on the skin
    :param active_dict: The collection of responses of real sensors tested on the e-skin
    :param max_len_trial: The minimum value for the length of one trial
    :param dataset: How the values of the pressure should be generated:
                    'real' : takes from the real data from a sample, if the sensor is not present in the sample the
                             algorithm randomly extract the signal from one of the other signal
                    'simulated' : just put the pressure value of the sensor to 25692250,
                    'noisy' : it's just simulated but with added noise (10% of 25692250)

    :return: The matrix of the excited sensors during the stimulus onset
    '''
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    sensors = my_skin.Position_to_Number([i for i in range(192)])
    # point = Point(0.5, 0.5)
    position_matrix = my_skin.give_position_matrix()
    polygon = Polygon(stimulus)
    # plt.plot([stimulus[i][0] for i in range(len(stimulus))] + [stimulus[0][0]],[stimulus[i][1] for i in range(len(stimulus))] + [stimulus[0][1]])
    excited_sensors = []
    if moving_texture == True:
        max_len_trial = 1
        dataset = 'noisy'
    for sensor in sensors:
        sensor_position = my_skin.find_sensor(sensor)
        if len(sensor_position[0]) > 0:
            row = sensor_position[0]
            column = sensor_position[1]
            # plt.plot(position_matrix[row, column, 0], position_matrix[row, column, 1], 'o', color='b')
            if polygon.contains(Point(position_matrix[row,column,0],position_matrix[row,column,1])):
                # chosen_trial = 25692250*np.ones([max_len_trial])
                # plt.plot(position_matrix[row, column, 0], position_matrix[row, column, 1], 'o', color='r')
                if dataset == 'real':
                    try:
                        key = my_skin.sensors_array[row,column]
                        possible_trials = active_dict[int(key)].keys()
                    except KeyError:
                        possible_keys = active_dict.keys()
                        key = np.random.choice(list(possible_keys))
                        possible_trials = active_dict[int(key)].keys()
                    chosen_trial = active_dict[int(key)][
                        np.random.choice(list(possible_trials))]
                    if (len(chosen_trial) > max_len_trial) & (moving_texture == False):
                        max_len_trial = len(chosen_trial)
                elif dataset == 'simulated':
                    chosen_trial = 25692250 * np.ones([max_len_trial])
                elif dataset == 'noisy':
                    chosen_trial = 25692250 * (np.ones([max_len_trial]) + noise * np.random.uniform(size = max_len_trial))
            else:
                chosen_trial = np.zeros([max_len_trial])
        else:
            chosen_trial = np.zeros([max_len_trial])
        excited_sensors.append(chosen_trial)
    for i in range(len(excited_sensors)):
        excited_sensors[i] = np.append(excited_sensors[i], np.zeros([max_len_trial - len(excited_sensors[i])]))
    excited_sensors = np.vstack(excited_sensors)
    return excited_sensors


def Generate_Sequence(my_skin,orientations, lengths, widths, analog_dictionary, ratio = '1:1', analog_clock = 10*ms,
                      spaces = None, numbers = None, depths = None, velocities = None, stimulus = 'bar',
                      trials_per_stimulus = 1, dataset = 'real', noise = None, progress_bar = True, parallelized = False):
    '''
    This function generates a sequence of stimuli applied on the artificial skin and simulate the response of the latter.
    At the current version the possibilities for 'stimulus' are:
    - bar: this option generates a bar applied on the skin with different orientations, lengths and widths.
    - texture: this option generates a texture (lot of bars in parallel) applied on the skin with different
               orientations, lengths, widths, spaces and number of bars.
    - sliding_texture: this option generate a texture that is sliding on the skin with different orientations, lengths,
               widths, spaces, number of bars an d velocities.
    The combination of different features (i.e. lengths, widths, ecc.) can be applied in a all:all combination (so
    length1xwidth1,length1xwidth2,...) or 1:1 combination (so length1xwidth1,length2xwidth2,...).
    Each combination is applied on the skin 1 time by default. The user can choose to apply many times changing the
    value of 'trials_per_stimulus'. The combinations are applied on the skin in a random series.
    The user can choose how the sensor responses are obtained:
     - 'real' takes them from an existing dataset, acquired in IIT, Italy.
     - 'simulated' generates them in a binary HIGH/LOW fashion
     - 'noisy' generates them like simulated but adding noise (in percentage to the HIGH value)
    :param my_skin:
    :param orientations:
    :param lengths:
    :param widths:
    :param analog_dictionary:
    :param ratio:
    :param analog_clock:
    :param spaces:
    :param numbers:
    :param depths:
    :param stimulus:
    :param trials_per_stimulus:
    :return:
    '''
    if len(analog_dictionary) > 0:
        active_trials = Find_Active_Trials(analog_dictionary)
    else:
        active_trials = []
    response = np.zeros([192,0])
    orientation_array = []
    time = 0*ms
    bar_array = []
    if (progress_bar == True) & (stimulus != 'sliding_texture'):
        print('At the moment the progress bar is available only for the sliding texture option')
    if stimulus == 'bar':
        if ratio == 'all:all':
            for orientation in orientations:
                for length in lengths:
                    for width in widths:
                        response_here = Generate_Response(stimulus=Create_Stimulus(my_skin = my_skin,orientation=orientation,
                                                                                   length=length, width=width),
                                                          active_dict=active_trials, my_skin = my_skin, dataset = dataset)
                        time += response_here.shape[1]*analog_clock
                        response = np.append(response,response_here, axis = 1)
                        orientation_array.append([time,[orientation,length,width]])
                        bar_array.append(Create_Stimulus(orientation=orientation, length=length, width=width))

        if ratio == '1:1':
            trials = []
            for i in range(len(orientations)):
                for j in range(trials_per_stimulus):
                    trials.append(i)
            trials = np.random.permutation(trials)
            for index in trials:
                response_here = Generate_Response(
                    stimulus=Create_Stimulus(my_skin = my_skin,orientation=orientations[index], length=lengths[index], width=widths[index]),
                    active_dict=active_trials, my_skin = my_skin, dataset = dataset)
                time += len(response_here) * analog_clock
                response = np.append(response,response_here, axis = 1)
                orientation_array.append([time, index])
        return response, orientation_array, bar_array
    elif stimulus == 'texture':
        texture_array = []
        if ratio == 'all:all':
            for orientation in orientations:
                for length in lengths:
                    for width in widths:
                        for space in spaces:
                            for number in numbers:
                                for depth in depths:
                                    response_here,geometry = Generate_Texture(my_skin = my_skin,
                                                                              active_trials = active_trials,
                                                                              length = length, width = width,
                                                                              space = space,number = number,
                                                                              depth = depth,
                                                                              orientation = orientation,
                                                                              geometry_coordinates= True)
                                    time += response_here.shape[1]*analog_clock
                                    response = np.append(response,response_here, axis = 1)
                                    orientation_array.append([time,[orientation,length,width,space,number,depth]])
                                    texture_array.append(geometry)

        if ratio == '1:1':
            len_array = [len(lengths),len(widths),len(spaces),len(numbers),len(depths), len(orientations)]
            max_len_array = np.max(len_array)
            for len_single in len_array:
                assert len_single == max_len_array, 'In 1:1 method all vectors should have same dimensions'
            trials = []
            for i in range(len(orientations)):
                for j in range(trials_per_stimulus):
                    trials.append(i)
            trials = np.random.permutation(trials)
            for index in trials:
                response_here, geometry = Generate_Texture(my_skin = my_skin, active_trials = active_trials,length=lengths[index], width=widths[index],
                                                 space=spaces[index], number=numbers[index],
                                                 depth=depths[index],
                                                 orientation=orientations[index],
                                                 geometry_coordinates = True)
                time += response_here.shape[1] * analog_clock
                response = np.append(response,response_here, axis = 1)
                orientation_array.append([time, index])
                texture_array.append(geometry)
        return response, orientation_array, texture_array
    elif stimulus == 'sliding_texture':
        texture_array = []
        stimulus_array = []
        positions = my_skin.give_position_matrix()
        x_length = positions[:, :, 0].max() - positions[:, :, 0].min()
        y_length = positions[:, :, 1].max() - positions[:, :, 1].min()
        total_length = np.sqrt(x_length ** 2 + y_length ** 2)
        if ratio == 'all:all':
            for orientation in orientations:
                for length in lengths:
                    for width in widths:
                        for space in spaces:
                            for number in numbers:
                                for depth in depths:
                                    for velocity in velocities:
                                        analog_clock = analog_clock/second
                                        step = my_skin.mm_to_pixels(velocity*analog_clock)

                                        iterations = 50*1/analog_clock
                                        for iter in range(iterations):
                                            response_here,geometry = Generate_Texture(my_skin = my_skin,
                                                                                      active_trials = active_trials,
                                                                                      length = length, width = width,
                                                                                      space = space,number = number,
                                                                                      depth = depth,
                                                                                      orientation = orientation,
                                                                                      geometry_coordinates= True,
                                                                                      texture_shift= step*iter,
                                                                                      moving_texture= True)
                                            time += response_here.shape[1]*analog_clock
                                            response = np.append(response,response_here, axis = 1)
                                        orientation_array.append([time,[orientation,length,width,space,number,depth]])
                                        texture_array.append(geometry)

        elif ratio == '1:1':
            len_array = [len(lengths),len(widths),len(spaces),len(numbers),len(depths), len(orientations)]
            max_len_array = np.max(len_array)
            for len_single in len_array:
                assert len_single == max_len_array, 'In 1:1 method all vectors should have same dimensions'
            trials = []
            for i in range(len(orientations)):
                for j in range(trials_per_stimulus):
                    trials.append(i)
            trials = np.random.permutation(trials)
            for tr_ix,index in enumerate(trials):
                bohh = analog_clock / second
                step = my_skin.mm_to_pixels(velocities[index] * bohh)
                iterations = int(np.floor(total_length/step))
                for iter in range(iterations):
                    if (progress_bar == True) & (parallelized == False):
                        Update_Progress((iter+tr_ix*iterations)/(iterations*len(trials)), 'Progress: ')
                    response_here, geometry = Generate_Texture(my_skin=my_skin,
                                                               active_trials=active_trials,
                                                               length=lengths[index], width=widths[index],
                                                               space=spaces[index], number=numbers[index],
                                                               depth=depths[index],
                                                               orientation=orientations[index],
                                                               geometry_coordinates=True,
                                                               texture_shift=step*iter - total_length/2 ,
                                                               moving_texture= True,
                                                               noise = noise)
                    time += response_here.shape[1] * analog_clock
                    response = np.append(response, response_here, axis=1)
                    texture_array.append(geometry)
                    stimulus_array.append([time, [0,0]])
                orientation_array.append([time, [velocities[index],orientations[index], lengths[index], widths[index], spaces[index],
                                                 numbers[index], depths[index]]])


        return response, orientation_array, texture_array, stimulus_array
def Find_Active_Trials(analog_dictionary):
    sensor_trials = {}
    counter = np.zeros([192])
    keys = analog_dictionary.keys()
    for key in keys:
        subkeys = analog_dictionary[key].keys()
        for subkey in subkeys:
            bbb = analog_dictionary[key][subkey]
            for row in range(bbb.shape[0]):
                aaa = bbb[row]
                actives = np.where(aaa > 25692250)
                if len(actives[0]) > 0.8 * len(aaa):
                    try:
                        sensor_trials[row][counter[row]] = {}
                        sensor_trials[row][counter[row]] = aaa
                        counter[row] += 1
                    except KeyError:
                        sensor_trials[row] = {}
                        sensor_trials[row][counter[row]] = aaa
                        counter[row] += 1
    return sensor_trials
def draw_Kiviat(seed,seed_id,information,total,statistic_collection = [], clustering_index = [], triangle_avg = [], polar_collection = []):
    # Code partially taken from https://python-graph-gallery.com/392-use-faceting-for-radar-chart/
    my_palette = plt.cm.get_cmap("Set2", 40)
    if len(statistic_collection) > 0:
        variables_n = len(statistic_collection)
    elif len(clustering_index) > 0:
        variables_n = len(clustering_index)
    elif len(triangle_avg) > 0:
        variables_n = len(triangle_avg)
    angles = [n / float(variables_n) * 2 * 3.14 for n in range(variables_n)]
    angles += angles[:1]
    rows = int(np.ceil(total/5))
    ax = plt.subplot(rows, 5, seed_id + 1, polar=True, )
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], ['RF' + str(i) for i in range(len(angles[:-1]))], color='grey', size=8)

    if len(statistic_collection) > 0:
        mean = []
        mean_hstd = []
        mean_lstd = []
        for rf in statistic_collection:
            mean.append(np.sqrt(rf['mean_x']**2 + rf['mean_y']**2))
            mean_hstd.append(np.sqrt(rf['mean_x']**2 + rf['mean_y']**2) + np.sqrt(rf['std_x']**2 + rf['std_y']**2))
            mean_lstd.append(np.sqrt(rf['mean_x'] ** 2 + rf['mean_y'] ** 2) - np.sqrt(rf['std_x'] ** 2 + rf['std_y'] ** 2))
        mean.append(np.sqrt(statistic_collection[0]['mean_x'] ** 2 + statistic_collection[0]['mean_y'] ** 2))
        mean_hstd.append(np.sqrt(statistic_collection[0]['mean_x']**2 + statistic_collection[0]['mean_y']**2) + np.sqrt(
            statistic_collection[0]['std_x']**2 + statistic_collection[0]['std_y']**2))
        mean_lstd.append(np.sqrt(statistic_collection[0]['mean_x'] ** 2 + statistic_collection[0]['mean_y'] ** 2) - np.sqrt(
            statistic_collection[0]['std_x'] ** 2 + statistic_collection[0]['std_y'] ** 2))
        ax.plot(angles, mean, color=my_palette(seed_id), linewidth=2, linestyle='solid')
        ax.fill(angles, mean, color=my_palette(seed_id), alpha=0.6)
        ax.fill(angles, mean_lstd, color=my_palette(seed_id), alpha=0.8)
        ax.fill(angles, mean_hstd, color=my_palette(seed_id), alpha=0.4)
    elif len(clustering_index) > 0:
        clustering_index += clustering_index[:1]
        ax.plot(angles, clustering_index, color=my_palette(seed_id), linewidth=2, linestyle='solid')
        ax.fill(angles, clustering_index, color=my_palette(seed_id), alpha=0.6)
        plt.yticks([0.01 * i for i in range(5)], [str(0.01 * i) for i in range(5)], color="grey", size=7)
        plt.ylim(0, 0.05)
    elif len(triangle_avg) > 0:
        triangle_avg += triangle_avg[:1]
        ax.plot(angles, triangle_avg, color=my_palette(seed_id), linewidth=2, linestyle='solid')
        ax.fill(angles, triangle_avg, color=my_palette(seed_id), alpha=0.6)
        plt.yticks([0.05*i for i in range(5)], [str(0.05*i) for i in range(5)], color="grey", size=7)
        plt.ylim(0, 0.2)
    elif len(polar_collection) > 0:
        polar_collection += polar_collection[:1]
        ax.plot(angles, polar_collection, color=my_palette(seed_id), linewidth=2, linestyle='solid')
        ax.fill(angles, polar_collection, color=my_palette(seed_id), alpha=0.6)
        plt.yticks([0.05*i for i in range(5)], [str(0.05*i) for i in range(5)], color="grey", size=7)
        plt.ylim(0, 0.2)

    plt.title("seed:" + str(seed) +  ". Info:" + str(round(information,2)) , size=11, color=my_palette(seed_id), y=1.1)
    # plt.show()
def create_folder(folder):
    import os
    try:
        os.mkdir(folder)
    except FileExistsError:
        while 1 == 0:
            print('we')
    return folder

def Cluster_Generator(my_skin, nNeurons_layer0,nNeurons_layer1,RF_n, sensors_for_RF, clustering_index_here=1, minimum_distance=100, increment_distance = 20, rings=10):
    if clustering_index_here == 0:
        clustering_index_here = 0.0001
    sensors = my_skin.Position_to_Number([i for i in range(192)])
    sensors_remaining = my_skin.Check_Position_Consinstency(sensors)
    w01 = np.zeros([nNeurons_layer0, nNeurons_layer1])
    # np.random.seed(5)
    sensor = -10
    centers_RF_collection = np.zeros([RF_n, sensors_for_RF])
    for row in range(RF_n):
        triangles = np.random.choice([i for i in range(len(my_skin.triangle_collection))], sensors_for_RF,
                                     replace=False)
        for column in range(sensors_for_RF):
            while sensor not in sensors_remaining:
                sensor = np.random.choice(my_skin.sensors_array[triangles[column]], 1, replace=False)
            centers_RF_collection[row, column] = sensor
            w01[my_skin.Number_to_Position(centers_RF_collection[row, column]), row] = 1
            sensors_remaining = np.setdiff1d(sensors_remaining, centers_RF_collection[row, column])

    # Plot_Skin_RF(my_skin, Image_PATH, Image_FILE, weights = w01, layer0 = nNeurons_layer0, layer1 = nNeurons_layer1, show_here= True, multiple_plots = False,
    #              save = folder_here + "/ReceptiveField_seed" + str(seed), )
    for column in range(sensors_for_RF):
        for row in range(RF_n):
            center = int(centers_RF_collection[row, column])
            for ring_distance in range(rings - 1):
                neighbour = my_skin.give_neighbour(sensors=center,
                                                   distance=[minimum_distance + increment_distance * (ring_distance),
                                                             minimum_distance + increment_distance * (ring_distance+1)])
                neighbour = np.intersect1d(neighbour, sensors_remaining)
                if len(neighbour) > 0:
                    if len(neighbour) > ring_distance:
                        sensors_chosen = np.random.choice(
                            neighbour.tolist() + [-1 for i in range(len(neighbour) * int(1 / clustering_index_here -1))],
                            ring_distance, replace=False)
                    else:
                        sensors_chosen = np.random.choice(
                            neighbour.tolist() + [-1 for i in range(len(neighbour) * int(1 / clustering_index_here -1))],
                            len(neighbour), replace=False)
                    sensors_chosen = sensors_chosen[np.where(sensors_chosen != -1)]
                    if len(sensors_chosen) != 0:
                        w01[my_skin.Number_to_Position(sensors_chosen), row] = 0.8
                        sensors_remaining = np.setdiff1d(sensors_remaining, sensors_chosen)
    return w01

def Find_Analog_Trials(value):
    avg = 0
    value_buffer = value
    for i in range(0, 192):
        for j in range(0, len(value[0])):
            if (value[i][j] < 25692250):
                value_buffer[i][j] = 0
    allsensors = np.sum(value_buffer, axis=0)
    startpoint = []
    endpoint = []
    i = 0
    while i < (len(allsensors)):
        if allsensors[i] != 0:
            for j in range(i, len(allsensors)):
                if allsensors[j] == 0:
                    for n in range(j, j + 10):
                        if allsensors[n] == 0:
                            avg += 1
                        if avg == 10:
                            endpoint.append(j)
                            startpoint.append(i)
                            i = j + 1

                    avg = 0
                    break
        i += 1

    orientations = ['-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90']
    allStimuli = {}
    ini = 0
    for key in orientations:

        for tri in range(5):
            trial = value[:, startpoint[ini] - 5:endpoint[ini] + 5]
            try:
                allStimuli[key][str(tri)] = trial
            except KeyError:
                allStimuli[key] = {}
                allStimuli[key][str(tri)] = trial
            ini += 1
    return allStimuli


def Shuffle_Analog_Data(orientations, simulation_time, seed, n_neurons, analog_clock):
    np.random.seed(seed)
    time = 0
    input_signal = np.zeros([n_neurons, 0])
    while time < simulation_time:
        key = np.random.choice(list(orientations.keys()))
        subkey = np.random.choice(list(orientations[key].keys()))
        input_signal = np.concatenate((input_signal, orientations[key][subkey]), axis=1)
        time += len(orientations[key][subkey].T) * analog_clock
    return input_signal


def Non_Shuffle_Analog_Data(orientations, simulation_time, seed, n_neurons, analog_clock):
    orientation_array = []
    keys_list = list(orientations.keys())
    np.random.seed(seed)
    time = 0 * second
    i = 0
    input_signal = np.zeros([n_neurons, 0])
    while time < simulation_time:
        key = keys_list[i]
        if i > 6:
            i = 0
        else:
            i += 1
        subkey = np.random.choice(list(orientations[key].keys()))
        input_signal = np.concatenate((input_signal, orientations[key][subkey]), axis=1)
        # print(time)
        # print(len(orientations[key][subkey].T))
        orientation_array.append([time, keys_list.index(key)])
        time += len(orientations[key][subkey].T) * analog_clock

    return input_signal, orientation_array


def Trials_Finder(pickle_data, trials_n):
    # print('Check point')
    # print(pickle_data)
    try:
        my_max = np.amax(pickle_data)
        timedif_or = my_max / trials_n
        timedif = timedif_or
        oreo_count = 0
        while oreo_count != trials_n:
            oreo_count = 1
            out = []
            last = []
            first = []
            timebefore = np.amin(pickle_data)
            first.append(timebefore)
            for time in pickle_data:
                if time > np.amin(pickle_data):
                    if (time - timebefore) > timedif:
                        # print("Value is higher")
                        first.append(time)
                        last.append(timebefore)
                        oreo_count += 1
                    timebefore = time
            if oreo_count > trials_n:
                timedif += timedif * 0.5
                # print("count is higher")
            if oreo_count < trials_n:
                timedif -= timedif * 0.5
            if timedif < timedif_or * 0.01:
                print("Low number of spikes, probably sporious. Ignoring them.")
                raise ValueError
        #print("Found count!")
        last.append(time)
        return first, last
    except ValueError:
        return [0], [0]


def Cancel_Double(first_coll, timedif):
    first_merged = list(itertools.chain(*first_coll))
    first_merged.sort()
    previous_first = 0
    shift = 0
    for ix, first in enumerate(first_merged[1:]):
        # print(first)
        # print(previous_first)
        if abs(first - previous_first) < timedif:
            del first_merged[ix - shift]
            shift += 1
        previous_first = first
    return first_merged


def popcol(my_array, pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    # pop = my_array[:,i]
    new_array = [np.array([]), np.array([]), np.array([])]
    new_array[0] = np.hstack((my_array[0][:i], my_array[0][i + 1:]))
    new_array[1] = np.hstack((my_array[1][:i], my_array[1][i + 1:]))
    new_array[2] = np.hstack((my_array[2][:i], my_array[2][i + 1:]))
    return new_array


def Orientation_Finder(pickle_data, orientations_n, trials_n):
    my_max = 0
    for row in pickle_data:
        here_max = np.amax(row)
        if here_max > my_max:
            my_max = np.amax(row)
    last_coll = []
    first_coll = []
    timedifmin = my_max
    for i_row, row in enumerate(pickle_data):
        timedif = my_max / orientations_n[i_row]
        # print(timedif)
        timedifmin = my_max / 8
        last = []
        first = []
        oreo_count = 0
        while oreo_count != orientations_n[i_row]:
            oreo_count = 1
            last = []
            first = []
            timebefore = np.amin(row)
            first.append(timebefore)
            for time in row:
                if time > np.amin(row):
                    if (time - timebefore) > timedif:
                        # print("Value is higher")
                        last.append(timebefore)
                        first.append(time)
                        oreo_count += 1
                    timebefore = time
            if oreo_count > orientations_n[i_row]:
                timedif += timedif * 0.5
                # print("count is higher")
            if oreo_count < orientations_n[i_row]:
                timedif -= timedif * 0.5
                # print("count is lower")
            # print("Oreo_count",oreo_count)
            # print(timedif)
        last.append(time)
        last_coll.append(last)
        first_coll.append(first)
    first_merged = Cancel_Double(first_coll, timedifmin * 0.5)
    last_merged = Cancel_Double(last_coll, timedifmin * 0.5)

    orientation_initial = []
    orientation_final = []
    # plt.figure()
    # axes1 = plt.axes()
    # for i,row in enumerate(pickle_data):
    #     axes1.scatter(row, [i] * len(spikes[i]))
    #     axes1.scatter(first_merged, [i] * len(first_merged))
    #     axes1.scatter(last_merged, [i] * len(last_merged))
    # plt.show()
    for index in range(0, len(first_merged)):
        initial_array = []
        final_array = []
        for row in range(0, len(pickle_data)):
            # print(row)
            # print(index)
            initial, final = Trials_Finder(pickle_data[row][np.bitwise_and(pickle_data[row] > first_merged[index],
                                                                           pickle_data[row] < last_merged[index])],
                                           trials_n)
            # print(str(row) + "," + str(len(initial)))
            # print(str(row) + "," + str(len(final)))
            initial_array.append(initial)
            final_array.append(final)
        # print("Orientation index " + str(index))
        # print(str(len(initial_array)) + "," + str([len(initial_array[i]) for i in range(len(initial_array))]))
        # print(str(len(final_array)) + "," + str([len(final_array[i]) for i in range(len(final_array))]))
        orientation_initial.append(initial_array)
        orientation_final.append(final_array)

    # plt.show()
    # print("EEE macarena")
    return orientation_initial, orientation_final


def extractSpikes(spikes, orientation_initial, orientation_final,
                  orientations=['-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90'],
                  colors=['Red', 'Green', 'Blue', 'Black', 'Gray']):
    # print('Eee macarena')
    # Each orientation has 3 neurons coming from three different receptive fields (gray,blue,black,...)
    # Each orientation is repeated 5 times, the numbers indicate the time interval in which they are presented
    allOreo = []
    for orientation in range(len(orientation_initial)):
        tmp2 = {}
        for rf in range(len(orientation_initial[orientation])):
            tmp = []
            for timestamp in range(len(orientation_initial[orientation][rf])):
                try:
                    tmp.append([orientation_initial[orientation][rf][timestamp],
                                orientation_final[orientation][rf][timestamp]])
                except:
                    pass
            if tmp != [[0, 0]]:
                tmp2.update({colors[rf]: tmp})
                # print(len(tmp))
                # print(str(orientations[orientation]) + "°, " +str(colors[rf]))
        allOreo.append(tmp2)

    orientation1 = {'Gray': [[0, 10], [10, 18], [18, 25], [25, 33], [33, 40]],
                    'Blue': [[0, 10], [10, 18], [18, 25], [25, 33], [33, 40]],
                    'Black': [[0, 10], [10, 18], [18, 25], [25, 33], [33, 40]]}
    orientation2 = {'Gray': [[40, 47], [47, 55], [55, 63], [63, 70], [70, 76]],
                    'Blue': [[40, 47], [47, 55], [55, 63], [63, 70], [70, 76]],
                    'Red': [[40, 47], [47, 55], [55, 63], [63, 70], [70, 76]]}
    orientation3 = {'Gray': [[77, 84], [84, 92], [92, 99], [99, 105], [105, 112]],
                    'Black': [[77, 85], [86, 94], [95, 103], 'MISSING', [110, 115]],
                    'Red': [[77, 84], [84, 92], [92, 99], [99, 105], [105, 112]]}
    orientation4 = {'Blue': [[117, 127], [127, 136], [136, 145], [145, 154], [154, 162]],
                    'Green': [[110, 118], [118, 128], [128, 136], [136, 145], [145, 153]],
                    'Red': [[117, 126], [126, 134], [134, 143], [143, 152], [152, 160]]}
    orientation5 = {'Black': [[160, 167], [167, 175], [175, 182], [182, 190], [190, 197]],
                    'Green': [[155, 162], [162, 170], [170, 177], [177, 185], [185, 192]],
                    'Red': [[160, 167], [167, 176], [176, 183], [183, 191], [191, 197]]}
    orientation6 = {'Gray': [[200, 206], [206, 215], [215, 220], [220, 226], [226, 233]],
                    'Green': [[200, 206], [206, 214], [214, 220], [220, 226], [226, 233]],
                    'Red': [[200, 207], [207, 213], [213, 219], [219, 225], 'MISSING']}
    orientation7 = {'Red': [[233, 240], [240, 247], [247, 255], [255, 262], [262, 268]],
                    'Blue': [[233, 240], [240, 247], [247, 255], [255, 262], [262, 268]],
                    'Black': [[233, 240], [240, 247], [247, 255], [255, 262], [262, 268]]}
    orientation8 = {'Blue': [[270, 275], [275, 280], [280, 288], [288, 295], [295, 305]],
                    'Gray': [[260, 267], [267, 273], [273, 280], [280, 287], [287, 295]],
                    'Green': [[270, 275], [275, 280], [280, 288], [288, 295], [295, 305]]}
    # allOreo = [orientation1,orientation2,orientation3,orientation4,orientation5,orientation6,orientation7,orientation8]
    # Here we give the different orientations
    allStimuli = {'-67.5': [], '-45': [], '-22.5': [], '0': [], '22.5': [], '45': [], '67.5': [], '90': []}
    # allStimuli = {'-67.5' : [],'-45' : [],'-22.5' : [],'0' : [],'22.5' : []}
    keys = list(allStimuli.keys())
    # Interation through Orientation
    for ori in range(len(allOreo)):
        # Choose one orientation
        oreo = allOreo[ori]
        oreoKeys = list(oreo.keys())
        for num in range(5):
            temp = []
            for color in range(3):
                # Here we take a receptive field from the spikes file
                try:
                    tempspikes = spikes[colors.index(oreoKeys[color])]
                except IndexError:
                    print("missing")
                    oreo[oreoKeys[color]][num] = 'MISSING'
                # Why oreo should have a missing element????
                try:
                    if oreo[oreoKeys[color]][num] == 'MISSING':
                        pass
                    else:
                        # For each time interval of the orientation presented, take the timestamps where the spikes happened
                        temptruth = [oreo[oreoKeys[color]][num][0] < x < oreo[oreoKeys[color]][num][1] for x in
                                     tempspikes]
                        temp.append([oreoKeys[color], tempspikes[temptruth]])
                    # In temp we find how many time a receptive field fired, for one orientation

                    # print(keys[ori])
                    # print(oreoKeys[color])
                    # print(temp)
                except IndexError:
                    print('Eee macarena')
            allStimuli[keys[ori]].append(temp)
    for key in keys:
        orin = allStimuli[key].copy()
        allStimuli[key] = []
        for stim in range(5):
            i = np.array([])
            t = np.array([])
            for x in range(3):
                try:
                    a = [orin[stim][j][1][0] for j in range(3)]
                    first = np.array([orin[stim][j][1][0] for j in range(3)])
                    idx = 0
                    difference = 0
                    diff = abs(first[x] - first)
                    #print(diff)
                    # ????????
                    if any(diff > 0.5):
                        idx = np.where(diff > 0.5)[0]
                        difference = np.round(first[x] - first[idx], 2)
                        for idxx in range(len(idx)):
                            orin[stim][idx[idxx]][1] = [
                                orin[stim][idx[idxx]][1][x] + difference[idxx] + np.random.uniform(0.003, 0.008) for x
                                in range(len(orin[stim][idx[idxx]][1]))]
                except IndexError:
                    print('Empty array')
            for col in range(3):
                color = orin[stim][col][0]
                spik = np.array(orin[stim][col][1])
                if color == 'Red':
                    i= np.concatenate((i,np.zeros(len(spik))))
                elif color == 'Green':
                    i= np.concatenate((i,np.ones(len(spik))*1))
                elif color == 'Blue':
                    i= np.concatenate((i,np.ones(len(spik))*2))
                elif color == 'Black':
                    i= np.concatenate((i,np.ones(len(spik))*3))
                else:
                    i= np.concatenate((i,np.ones(len(spik))*4))
                t = np.concatenate((t,spik))
            allStimuli[key].append([i,t])
    return allStimuli
def shuffle(stimuli,SimTime,timeBetween):
    np.random.seed(2)
    everyDayImShuffling = [np.array([]),np.array([]),np.array([])]
    orientation_array = []
    numOfStims = 400
    repPerStim = 5
    currentTime = 1
    keys = list(stimuli.keys())
    while currentTime < SimTime:

        if len(keys)>0:
            pickOri = np.random.randint(0,len(stimuli))
            #print(pickOri)
            ori = stimuli[keys[pickOri]]
            pickStim = np.random.randint(0,len(ori))
            #print(pickStim)
            stim = ori[pickStim]
            diffToTime = currentTime - stim[1].min()
            everyDayImShuffling[0] = np.concatenate((everyDayImShuffling[0],stim[0]))
            everyDayImShuffling[1] = np.concatenate((everyDayImShuffling[1],stim[1]+diffToTime))
            everyDayImShuffling[2] = np.concatenate((everyDayImShuffling[2], [pickOri for i in range(0,len(stim[0]))]))
            orientation_array.append([stim[1].max()+diffToTime,pickOri])
            #print(everyDayImShuffling[1].max())
            currentTime = everyDayImShuffling[1].max()+timeBetween
        #print(currentTime)
    shift = 0
    prev_timestamps = 0
    sort_indexes = np.argsort(everyDayImShuffling[1])
    new_stimuli = [np.array([]), np.array([]), np.array([])]
    new_stimuli[1] = everyDayImShuffling[1][sort_indexes]
    new_stimuli[0] = everyDayImShuffling[0][sort_indexes]
    new_stimuli[2] = everyDayImShuffling[2][sort_indexes]

    for index in range(0, len(new_stimuli[1])):
        new_stimuli[1][index - shift] = round(new_stimuli[1][index - shift], 3)
        # print(abs(stimuli[1][index] - prev_timestamps))
        if abs(new_stimuli[1][index - shift] - prev_timestamps) < 1E-3:
            #print("meno di un ms")
            #print(str(new_stimuli[1][index - shift]) + "," + str(prev_timestamps))
            new_stimuli = popcol(new_stimuli, index - shift)
            shift += 1
        else:
            prev_timestamps = new_stimuli[1][index - shift]


def shuffle_parametric(stimuli,SimTime,seed,timeBetween):
    np.random.seed(seed)
    everyDayImShuffling = [np.array([]),np.array([]),np.array([])]
    orientation_array = []
    numOfStims = 400
    repPerStim = 5
    currentTime = 1
    keys = list(stimuli.keys())
    while currentTime < SimTime:

        if len(keys)>0:
            pickOri = np.random.randint(0,len(stimuli))
            #print(pickOri)
            ori = stimuli[keys[pickOri]]
            pickStim = np.random.randint(0,len(ori))
            #print(pickStim)
            stim = ori[pickStim]
            diffToTime = currentTime - stim[1].min()
            everyDayImShuffling[0] = np.concatenate((everyDayImShuffling[0],stim[0]))
            everyDayImShuffling[1] = np.concatenate((everyDayImShuffling[1],stim[1]+diffToTime))
            everyDayImShuffling[2] = np.concatenate((everyDayImShuffling[2], [pickOri for i in range(0,len(stim[0]))]))
            orientation_array.append([stim[1].max()+diffToTime,pickOri])
            #print(everyDayImShuffling[1].max())
            currentTime = everyDayImShuffling[1].max()+timeBetween
        #print(currentTime)
    shift = 0
    prev_timestamps = 0
    sort_indexes = np.argsort(everyDayImShuffling[1])
    new_stimuli = [np.array([]), np.array([]), np.array([])]
    new_stimuli[1] = everyDayImShuffling[1][sort_indexes]
    new_stimuli[0] = everyDayImShuffling[0][sort_indexes]
    new_stimuli[2] = everyDayImShuffling[2][sort_indexes]

    for index in range(0, len(new_stimuli[1])):
        new_stimuli[1][index - shift] = round(new_stimuli[1][index - shift], 3)
        # print(abs(stimuli[1][index] - prev_timestamps))
        if abs(new_stimuli[1][index - shift] - prev_timestamps) < 1E-3:
            #print("meno di un ms")
            #print(str(new_stimuli[1][index - shift]) + "," + str(prev_timestamps))
            new_stimuli = popcol(new_stimuli, index - shift)
            shift += 1
        else:
            prev_timestamps = new_stimuli[1][index - shift]

    return new_stimuli, orientation_array
def non_shuffle(stimuli,SimTime,timeBetween):
    np.random.seed(2)
    everyDayImShuffling = [np.array([]),np.array([]),np.array([])]
    orientation_array = []
    numOfStims = 400
    repPerStim = 1
    currentTime = 1
    keys = list(stimuli.keys())
    index = 0
    pickOri = 0
    while currentTime < SimTime:

        if len(keys)>0:
            if index % repPerStim == 0:
                pickOri += 1
            if pickOri == 8:
                pickOri = 0
            #pickOri = np.random.randint(0,len(stimuli))
            #print(pickOri)
            ori = stimuli[keys[pickOri]]
            pickStim = np.random.randint(0,len(ori))
            #print(pickStim)
            stim = ori[pickStim]
            diffToTime = currentTime - stim[1].min()
            everyDayImShuffling[0] = np.concatenate((everyDayImShuffling[0],stim[0]))
            everyDayImShuffling[1] = np.concatenate((everyDayImShuffling[1],stim[1]+diffToTime))
            everyDayImShuffling[2] = np.concatenate((everyDayImShuffling[2], [pickOri for i in range(0,len(stim[0]))]))
            orientation_array.append([stim[1].max()+diffToTime,pickOri])
            #print(everyDayImShuffling[1].max())
            currentTime = everyDayImShuffling[1].max()+timeBetween
            index += 1
        #print(currentTime)
    shift = 0
    prev_timestamps = 0
    sort_indexes = np.argsort(everyDayImShuffling[1])
    new_stimuli = [np.array([]), np.array([]), np.array([])]
    new_stimuli[1] = everyDayImShuffling[1][sort_indexes]
    new_stimuli[0] = everyDayImShuffling[0][sort_indexes]
    new_stimuli[2] = everyDayImShuffling[2][sort_indexes]

    for index in range(0, len(new_stimuli[1])):
        new_stimuli[1][index - shift] = round(new_stimuli[1][index - shift], 3)
        # print(abs(stimuli[1][index] - prev_timestamps))
        if abs(new_stimuli[1][index - shift] - prev_timestamps) < 1E-3:
            #print("meno di un ms")
            #print(str(new_stimuli[1][index - shift]) + "," + str(prev_timestamps))
            new_stimuli = popcol(new_stimuli, index - shift)
            shift += 1
        else:
            prev_timestamps = new_stimuli[1][index - shift]

    return new_stimuli, orientation_array
def non_shuffle_parametric(stimuli,SimTime,seed,timeBetween):
    '''

    :param stimuli:
    :param SimTime:
    :param seed:
    :param timeBetween:
    :return:
    '''
    np.random.seed(seed)
    everyDayImShuffling = [np.array([]),np.array([]),np.array([])]
    orientation_array = []
    numOfStims = 400
    repPerStim = 1
    currentTime = 1
    keys = list(stimuli.keys())
    index = 0
    pickOri = 0
    while currentTime < SimTime:

        if len(keys)>0:
            if index % repPerStim == 0:
                pickOri += 1
            if pickOri == 8:
                pickOri = 0
            #pickOri = np.random.randint(0,len(stimuli))
            #print(pickOri)
            ori = stimuli[keys[pickOri]]
            pickStim = np.random.randint(0,len(ori))
            #print(pickStim)
            stim = ori[pickStim]
            diffToTime = currentTime - stim[1].min()
            everyDayImShuffling[0] = np.concatenate((everyDayImShuffling[0],stim[0]))
            everyDayImShuffling[1] = np.concatenate((everyDayImShuffling[1],stim[1]+diffToTime))
            everyDayImShuffling[2] = np.concatenate((everyDayImShuffling[2], [pickOri for i in range(0,len(stim[0]))]))
            orientation_array.append([stim[1].max()+diffToTime,pickOri])
            #print(everyDayImShuffling[1].max())
            currentTime = everyDayImShuffling[1].max()+timeBetween
            index += 1
        #print(currentTime)
    shift = 0
    prev_timestamps = 0
    sort_indexes = np.argsort(everyDayImShuffling[1])
    new_stimuli = [np.array([]), np.array([]), np.array([])]
    new_stimuli[1] = everyDayImShuffling[1][sort_indexes]
    new_stimuli[0] = everyDayImShuffling[0][sort_indexes]
    new_stimuli[2] = everyDayImShuffling[2][sort_indexes]

    for index in range(0, len(new_stimuli[1])):
        new_stimuli[1][index - shift] = round(new_stimuli[1][index - shift], 3)
        # print(abs(stimuli[1][index] - prev_timestamps))
        if abs(new_stimuli[1][index - shift] - prev_timestamps) < 1E-3:
            #print("meno di un ms")
            #print(str(new_stimuli[1][index - shift]) + "," + str(prev_timestamps))
            new_stimuli = popcol(new_stimuli, index - shift)
            shift += 1
        else:
            prev_timestamps = new_stimuli[1][index - shift]

    return new_stimuli, orientation_array
#@implementation('numpy', discard_units=True)
# allParam_testing = dict()
# orientation_array2 = np.array([-1,-20,55])
#
# def orientation_array_hanlder():
#     stimuli,orientation_array = shuffle(spikes_sorted, 100)
#     orientation_array2 = np.array([orientation_array[i][0] for i in range(0,len(orientation_array))])
#     return orientation_array2
# @implementation('cpp', r'''float ascarambad(int index){
#                         for(int i= 0 ; i<3;i++){
#                         //cout << "orientation_array: " << _namespaceorientation_array[index] << " index: " << index  << endl;
#                         }
#                         return _namespaceorientation_array[index];
#                         }
#                         ''',namespace={"orientation_array": orientation_array_hanlder()})
#
#
# @check_units(t=second, orientation_array=second, index=1, result=1)
# def ascarambad(index):
#     return #orientation_array[index];