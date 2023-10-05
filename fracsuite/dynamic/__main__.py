from fracsuite.splinters.analyzer import AnalyzerConfig

config = AnalyzerConfig()

print(config.yaml())



# # implement parse to make this script callable from outside
# parser = argparse.ArgumentParser()    
# parser.add_argument('-image', nargs="?", help='The image to be processed.')
# parser.add_argument('-realsize', nargs=2, help='Real size of the input image. Use: -realsize W H',\
#     type=int)
# parser.add_argument('-plot-ext', nargs="?", help='Plot file extension. Default: png.', \
#     default="png", choices=['png', 'pdf', 'jpg', 'bmp'])
# parser.add_argument('-image-ext', nargs="?", help='Image file extension. Default: png.',\
#     default="png", choices=['png', 'jpg', 'bmp'])
# parser.add_argument('--crop', action='store_true', \
#     help='Instruct the analyzer to crop the input image.', default=False)
# parser.add_argument('--displayplots', action='store_true', \
#     help='Instruct the analyzer to display output plots.', default=False)

# args = parser.parse_args()    

# print(args)

# if args.realsize is not None:
#     args.realsize = tuple(args.realsize)
    
# analyzer = Analyzer(args.image, args.crop, img_real_size=args.realsize)

# analyzer.plot(display=args.displayplots)
# analyzer.plot_area(display=args.displayplots)
# analyzer.plot_area_2(display=args.displayplots)

# analyzer.save_images(extension=args.image_ext)
# analyzer.save_plots(extension=args.plot_ext)