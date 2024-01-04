import os 
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt


#relative path setting

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
os.chdir(path)


df = pd.read_excel("volume-data.xlsx")
df

two_wheelers = df['Frequency'][0]
four_wheelers = df['Frequency'][1]
Bus_Truck = (df['Frequency'][2] + df['Frequency'][3])

PCU2W = two_wheelers * 0.75
PCU4W = four_wheelers * 1
PCUHW = Bus_Truck * 2.8
print(PCU2W)
print(PCU4W)
print(PCUHW)

total = PCU2W + PCU4W + PCUHW
Time = 47/3600
volume = total/Time
print(volume)


labels = ['Two-Wheeler', 'Four-Wheeler', 'Bus/Truck']
sizes = [PCU2W, four_wheelers, Bus_Truck]

plt.pie(sizes,autopct='%.2f',shadow=True)
plt.legend(labels)
plt.savefig("pie.png", format="png", bbox_inches="tight")


pdf = FPDF()
pdf.add_page()

pdf.set_font('Arial','BU',30)
pdf.cell(0,130,ln = 1)
pdf.cell(0,20, 'TRAFFIC  ANALYSIS  REPORT',align = 'C',border = 1,ln = 1 )

pdf.add_page()
pdf.set_font('Arial','BU',20)
pdf.cell(0, 20, 'VOLUME STUDY',align = 'C',border = 1,ln = 1 )

pdf.cell(0,20, ln = 1)
pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total 2 Wheelers = ' + str(int(two_wheelers)) ,ln = 0 ,border = 0)

pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total Impact in terms of PCU = ' + str(PCU2W) ,ln = 1 ,border = 0)


pdf.cell(0,5, ln = 1)
pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total 3 Wheelers = ' + str(0) ,ln = 0 , border = 0)

pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total Impact in terms of PCU = ' + str(0) ,ln = 1 ,border = 0)

pdf.cell(0,5, ln = 1)
pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total 4 Wheelers (Cars) = ' + str(int(four_wheelers)) ,ln = 0 ,border = 0)

pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total Impact in terms of PCU = ' + str(PCU4W) ,ln = 1 ,border = 0)

pdf.cell(0,5, ln = 1)
pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total Bus/Trucks = ' + str(int(Bus_Truck)) ,ln = 0 , border = 0)

pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Total Impact in terms of PCU = ' + str(round(PCUHW,2)) ,ln = 1 ,border = 0)

pdf.cell(0,10,ln = 1)
pdf.set_font('Arial','B',15)
pdf.cell(100, 10, 'Traffic Volume = ' + str(round(volume,2)) + ' pcu/hr' ,ln = 1 ,border = 0)

pdf.cell(0,15,ln = 1)
pdf.set_font('Arial','BUI',18)
pdf.cell(0,10,'PLOTS :-',ln = 1)
pdf.cell(0,5,ln = 1)
pdf.cell(45,10,ln = 0)

pdf.image('pie.png', x = None, y = None, w = 120, h = 120, type = '', link = '',)

pdf.output(r'result_v.pdf', 'F')
