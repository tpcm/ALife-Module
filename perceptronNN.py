import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def step(x):
    if x > 0:
        y = 1
        return y
    else:
        y = 0
        return y

class PerceptronNN():

    def __init__(self, genotype, smell_acc, Lrate):
        
        self.genotype = genotype
        self.smell_acc = smell_acc
        self.Lrate = Lrate

        # Convert genotype into network
        # Genotype 0 - 23 specify input node types
        # Convert genotype list entries into integers
        # Then assign unit 

        # Determine if network has unit A
        on_off1 = int("".join(str(x) for x in self.genotype[0:3]), 2)
        if on_off1 <= 3:
            self.x1_on = False
        else:
            self.x1_on = True

        # Determine unit A type
        unit1 = int("".join(str(x) for x in self.genotype[4:7]), 2)
        if self.genotype[3] == 0:
            if unit1 <=1:
                self.x1 = "sweet"
            elif unit1 > 1 and unit1 <= 3:
                self.x1 = "sour"
            elif unit1 > 3 and unit1 <= 5:
                self.x1 = "red"
            else:
                self.x1 = "green"
        else:
            if unit1 <=3:
                self.x1 = "hidden"
            else:
                self.x1 = "motor"
        
        # Determine if netowrk has unit B
        on_off2 = int("".join(str(x) for x in self.genotype[7:10]), 2)
        if on_off2 <= 3:
            self.x2_on = False
        else:
            self.x2_on = True

        # Determine unit B type
        unit2 = int("".join(str(x) for x in self.genotype[11:14]), 2)
        if self.genotype[10] == 0:
            if unit2 <=1:
                self.x2 = "sweet"
            elif unit2 > 1 and unit2 <= 3:
                self.x2 = "sour"
            elif unit2 > 3 and unit2 <= 5:
                self.x2 = "red"
            else:
                self.x2 = "green"
        else:
            if unit2 <=3:
                self.x2 = "hidden"
            else:
                self.x2 = "motor"
        
        # Determine if network has unit C
        on_off3 = int("".join(str(x) for x in self.genotype[14:17]), 2)
        if on_off3 <= 3:
            self.x3_on = False
        else:
            self.x3_on = True

        # Determine unit C type
        unit3 = int("".join(str(x) for x in self.genotype[18:21]), 2)
        if self.genotype[17] == 0:
            if unit3 <=1:
                self.x3 = "sweet"
            elif unit3 > 1 and unit2 <= 3:
                self.x3 = "sour"
            elif unit3 > 3 and unit2 <= 5:
                self.x3 = "red"
            else:
                self.x3 = "green"
        else:
            if unit3 <=3:
                self.x3 = "hidden"
            else:
                self.x3 = "motor"
        
        # Determine whether network has a connection between A and C
        conn1 = int("".join(str(x) for x in self.genotype[21:24]), 2)
        if conn1 <= 3:
            self.w1_on = False
        else:
            self.w1_on = True
        # If network has connection A-C, set weight and determine plasticity
        if self.w1_on == True:
            weight1 = int("".join(str(x) for x in self.genotype[25:28]), 2)
            # Check previous bit to see if weight is positive
            if self.genotype[24] == 0:
                if weight1 <= 1:
                    self.w1 = 0
                elif weight1 > 1 and weight1 <= 3:
                    self.w1 = 1
                elif weight1 > 3 and weight1 <= 5:
                    self.w1 = 2
                else:
                    self.w1 = 3
            # Check previous bit to see if weight is negative
            else:
                if weight1 <= 1:
                    self.w1 = 0
                elif weight1 > 1 and weight1 <= 3:
                    self.w1 = -1
                elif weight1 > 3 and weight1 <= 5:
                    self.w1 = -2
                else:
                    self.w1 = -3
            # Check plasticity
            if self.genotype[28] == 0:
                self.w1_learnable = True
            else:
                self.w1_learnable = False
        # If network has no connection A-C, set weight to zero and fix weight
        else:
            self.w1 = 0
            self.w1_learnable = False

        # Determine whether network has a connection between A and B
        conn2 = int("".join(str(x) for x in self.genotype[29:32]), 2)
        if conn2 <= 3:
            self.w2_on = False
        else:
            self.w2_on = True
        # If network has connection A-B, set weight and determine plasticity
        if self.w2_on == True:
            weight2 = int("".join(str(x) for x in self.genotype[33:36]), 2)
            # Check previous bit to see if weight is positive
            if self.genotype[32] == 0:
                if weight2 <= 1:
                    self.w2 = 0
                elif weight2 > 1 and weight2 <= 3:
                    self.w2 = 1
                elif weight2 > 3 and weight2 <= 5:
                    self.w2 = 2
                else:
                    self.w2 = 3
            # Check previous bit to see if weight is negative
            else:
                if weight2 <= 1:
                    self.w2 = 0
                elif weight2 > 1 and weight2 <= 3:
                    self.w2 = -1
                elif weight2 > 3 and weight2 <= 5:
                    self.w2 = -2
                else:
                    self.w2 = -3
            # Check plasticity
            if self.genotype[36] == 0:
                self.w2_learnable = True
            else:
                self.w2_learnable = False
        # If network has no connection A-B, set weight to zero and fix weight
        else:
            self.w2 = 0
            self.w2_learnable = False

        # Determine whether network has a connection between B and C
        conn3 = int("".join(str(x) for x in self.genotype[37:40]), 2)
        if conn3 <= 3:
            self.w3_on = False
        else:
            self.w3_on = True
        # If network has connection B-C, set weight and determine plasticity
        if self.w3_on == True:
            weight3 = int("".join(str(x) for x in self.genotype[41:44]), 2)
            # Check previous bit to see if weight is positive
            if self.genotype[40] == 0:
                if weight3 <= 1:
                    self.w3 = 0
                elif weight3 > 1 and weight3 <= 3:
                    self.w3 = 1
                elif weight3 > 3 and weight3 <= 5:
                    self.w3 = 2
                else:
                    self.w3 = 3
            # Check previous bit to see if weight is negative
            else:
                if weight3 <= 1:
                    self.w3 = 0
                elif weight3 > 1 and weight3 <= 3:
                    self.w3 = -1
                elif weight3 > 3 and weight3 <= 5:
                    self.w3 = -2
                else:
                    self.w3 = -3
            # Check plasticity
            if self.genotype[44] == 0:
                self.w3_learnable = True
            else:
                self.w3_learnable = False
        # If network has no connection B-C, set weight to zero and fix weight
        else:
            self.w3 = 0
            self.w3_learnable = False
        
        # Determine whether network has bias1
        bon1 = int("".join(str(x) for x in self.genotype[45:48]), 2)
        if bon1 <= 3:
            self.b1_on = False
        else:
            self.b1_on = True
        # If network has b1, determine value
        if self.b1_on == True:
            bias1 = int("".join(str(x) for x in self.genotype[49:52]), 2)
            # Check previous bit to see if bias is positive
            if self.genotype[48] == 0:
                if bias1 <= 1:
                    self.b1 = 0
                elif bias1 > 1 and bias1 <= 3:
                    self.b1 = 1
                elif bias1 > 3 and bias1 <= 5:
                    self.b1 = 2
                else:
                    self.b1 = 3
            # Check previous bit to see if bias is negative
            else:
                if bias1 <= 1:
                    self.b1 = 0
                elif bias1 > 1 and bias1 <= 3:
                    self.b1 = -1
                elif bias1 > 3 and bias1 <= 5:
                    self.b1 = -2
                else:
                    self.b1 = -3
        # If network does not have bias fot unit A, set bias to 0
        else:
            self.b1 = 0

        # Determine whether network has bias for unit B
        bon2 = int("".join(str(x) for x in self.genotype[52:55]), 2)
        if bon2 <= 3:
            self.b2_on = False
        else:
            self.b2_on = True
        # If network has b2, determine value
        if self.b2_on == True:
            bias2 = int("".join(str(x) for x in self.genotype[56:59]), 2)
            # Check previous bit to see if bias is positive
            if self.genotype[55] == 0:
                if bias2 <= 1:
                    self.b2 = 0
                elif bias2 > 1 and bias2 <= 3:
                    self.b2 = 1
                elif bias2 > 3 and bias2 <= 5:
                    self.b2 = 2
                else:
                    self.b2 = 3
            # Check previous bit to see if bias is negative
            else:
                if bias2 <= 1:
                    self.b2 = 0
                elif bias2 > 1 and bias2 <= 3:
                    self.b2 = -1
                elif bias2 > 3 and bias2 <= 5:
                    self.b2 = -2
                else:
                    self.b2 = -3
        # If network does not have bias for unit B, set b2 to 0
        else:
            self.b2 = 0

        # Determine whether network has bias for unit C
        bon3 = int("".join(str(x) for x in self.genotype[59:62]), 2)
        if bon3 <= 3:
            self.b3_on = False
        else:
            self.b3_on = True
        # If network has b3, determine value
        if self.b3_on == True:
            bias3 = int("".join(str(x) for x in self.genotype[63:]), 2)
            # Check previous bit to see if bias is positive
            if self.genotype[62] == 0:
                if bias3 <= 1:
                    self.b3 = 0
                elif bias3 > 1 and bias3 <= 3:
                    self.b3 = 1
                elif bias3 > 3 and bias3 <= 5:
                    self.b3 = 2
                else:
                    self.b3 = 3
            # Check previous bit to see if bias is negative
            else:
                if bias3 <= 1:
                    self.b3 = 0
                elif bias3 > 1 and bias3 <= 3:
                    self.b3 = -1
                elif bias3 > 3 and bias3 <= 5:
                    self.b3 = -2
                else:
                    self.b3 = -3
        # If network does not have bias for unit C, set b3 to 0
        else:
            self.b3 = 0

        
        self.x1_activation = 0
        self.x2_activation = 0
        self.x3_activation = 0
        self.x2_input = 0
        self.x3_input = 0
        self.output = 0

    def feedForward(self, sw_or_so, r_or_g):
        """
            Feed Forward algorithm for 1st environment in which food is red and poison is green
            :param substance: 1-D array containing 0 or 1 in both entries
                                If sw_or_so = 0 then sweet
                                If sw_or_so = 1 then sour
                                if r_or_g = 0 then red
                                if r_or_g = 1 then green
        """
        
        if self.x1_on == True:
            # If x1 is sweet smelling unit, and substance smells sweet
            if self.x1 == "sweet" and sw_or_so == 0:
                self.x1_activation = 1
            # If x1 is sweet smelling unit, and substance smells sour
            elif self.x1 == "sweet" and sw_or_so == 1:
                self.x1_activation = 0
            # If x1 is sour smelling unit, and substance smells sweet
            elif self.x1 == "sour" and sw_or_so == 0:
                self.x1_activation = 0
            # If x1 is sour smelling unit, and substance smells sour
            elif self.x1 == "sour" and sw_or_so == 1:
                self.x1_activation = 1
            # If x1 is red unit, and subastance is red
            elif self.x1 == "red" and r_or_g == 0:
                self.x1_activation = 1
            # If x1 is red unit, and substance is green
            elif self.x1 == "red" and r_or_g == 1:
                self.x1_activation = 0
            # If x1 is green unit and substance is red
            elif self.x1 == "green" and r_or_g == 0:
                self.x1_activation = 0
            # If x1 is green unit and substance is green
            elif self.x1 == "green" and r_or_g == 1:
                self.x1_activation = 1
            # If x1 is a hidden unit, can't take input
            elif self.x1 == "hidden":
                self.x1_activation = 0
            # If x1 is output unit, can't take input
            elif self.x1 == "motor":
                self.x1_activation = 0
                return 0
        # If network doesn't have x1 unit
        else:
            self.x1_activation = 0
        
        self.x2_input = self.w2 * self.x1_activation

        # If network has unit B
        if self.x2_on == True:
            # If x2 is sweet smelling unit, and substance smells sweet
            if self.x2 == "sweet" and sw_or_so == 0:
                self.x2_activation = step(1 + self.x2_input)
            # If x2 is sweet smelling unit, and substance smells sour
            elif self.x2 == "sweet" and sw_or_so == 1:
                self.x2_activation =step(0 + self.x2_input)
            # If x2 is sour smelling unit, and substance smells sweet
            elif self.x2 == "sour" and sw_or_so == 0:
                self.x2_activation = step(0 + self.x2_input)
            # If x2 is sour smelling unit, and substance smells sour
            elif self.x2 == "sour" and sw_or_so == 1:
                self.x2_activation = step(1 + self.x2_input)
            # If x2 is red unit, and subastance is red
            elif self.x2 == "red" and r_or_g == 0:
                self.x2_activation = step(1 + self.x2_input)
            # If x2 is red unit, and substance is green
            elif self.x2 == "red" and r_or_g == 1:
                self.x2_activation = step(0 + self.x2_input)
            # If x2 is green unit and substance is red
            elif self.x2 == "green" and r_or_g == 0:
                self.x2_activation = step(0 + self.x2_input)
            # If x2 is green unit and substance is green
            elif self.x2 == "green" and r_or_g == 1:
                self.x2_activation = step(1 + self.x2_input)
            # If x2 is a hidden unit, can't take input
            elif self.x2 == "hidden":
                self.x2_activation = sigmoid(self.x2_input)
            # If x2 is output unit, can't take input
            elif self.x2 == "motor":
                self.x2_activation = np.sign(self.x2_input)
                return self.x2_activation
            
        # If network doesn't have x2 unit
        else:
            self.x2_activation = 0

        self.x3_input = (self.w1 * self.x1_activation) + (self.w3 * self.x2_activation)

        # If network has unit C
        if self.x3_on == True:
            # If x2 is sweet smelling unit, and substance smells sweet
            if self.x3 == "sweet" and sw_or_so == 0:
                self.x3_activation = step(1 + self.x3_input)
            # If x2 is sweet smelling unit, and substance smells sour
            elif self.x3 == "sweet" and sw_or_so == 1:
                self.x3_activation =step(0 + self.x3_input)
            # If x2 is sour smelling unit, and substance smells sweet
            elif self.x3 == "sour" and sw_or_so == 0:
                self.x3_activation = step(0 + self.x3_input)
            # If x2 is sour smelling unit, and substance smells sour
            elif self.x3 == "sour" and sw_or_so == 1:
                self.x3_activation = step(1 + self.x3_input)
            # If x2 is red unit, and subastance is red
            elif self.x3 == "red" and r_or_g == 0:
                self.x3_activation = step(1 + self.x3_input)
            # If x2 is red unit, and substance is green
            elif self.x3 == "red" and r_or_g == 1:
                self.x3_activation = step(0 + self.x3_input)
            # If x2 is green unit and substance is red
            elif self.x3 == "green" and r_or_g == 0:
                self.x3_activation = step(0 + self.x3_input)
            # If x2 is green unit and substance is green
            elif self.x3 == "green" and r_or_g == 1:
                self.x3_activation = step(1 + self.x3_input)
            # If x2 is a hidden unit, can't take input
            elif self.x3 == "hidden":
                self.x3_activation = sigmoid(self.x3_input)
            # If x2 is output unit, can't take input
            elif self.x3 == "motor":
                self.x3_activation = np.sign(self.x3_input)
                return self.x3_activation
            
        # If network doesn't have x2 unit
        else:
            self.x3_activation = 0
        # If network has no output neurons it returns 0
        return 0

    def updateWeights(self):
        """
            A method to update the weights of learnable connections
            :return N/A:
        """
        if self.w1_learnable and self.w1 < 3:
            self.w1 += self.Lrate * self.x1_activation * self.x3_activation
        if self.w2_learnable and self.w2 < 3:
            self.w2 += self.Lrate * self.x1_activation * self.x2_activation
        if self.w3_learnable and self.w3 < 3:
            self.w3 += self.Lrate * self.x2_activation * self.x3_activation

def main():
    # Testing
    genotype = [1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    trial = PerceptronNN(genotype, 0.75, 1)
    print(f"{trial.w1}")
    print(f"{trial.w2}")
    print(f"{trial.w3}")
    x = trial.feedForward(0,0)
    print(f"{trial.x1_activation}")
    print(f"{trial.x2_activation}")
    print(f"{trial.x3_activation}")
    trial.updateWeights()
    print(f"{trial.w1}")
    print(f"{trial.w2}")
    print(f"{trial.w3}")
    x = trial.feedForward(0,0)
    print(f"{trial.x1_activation}")
    print(f"{trial.x2_activation}")
    print(f"{trial.x3_activation}")


if __name__ == "__main__":
    main()