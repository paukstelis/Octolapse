import octoprint_octolapse.utility as utility

from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np

class FrameCompare(object):

	def __init__(self,octoprint_printer,data_folder,starttime):
		self.OctoprintPrinter = octoprint_printer
		self.scores = []

		#Things that will eventually come from settings
		
		#When to start counting,NOTE: currently still getting snap count after increment
		self.start_count = 3
		#The number of std dev over mean that triggers a message
		self.sd_over_thresh = 3.0
		#The number of std dev over mean that indicate the frame itself may be bad.
		self.sd_badframe_thresh = 15.0
		#The number of snaps to wait before getting/adding to statistics
		self.begin_analysis_count = 3

	def _read_image(self, data_folder, starttime, snapcount):
		printing =  utility.get_currently_printing_filename(self.OctoprintPrinter)
		snapshot_path = utility.get_snapshot_temp_directory(data_folder)
		imagefile = utility.get_snapshot_filename(printing, starttime, snapcount)
		path = ("{0}{1}".format(snapshot_path,imagefile))

		image = cv2.imread(path)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return image, path

	def _to_gray(self, image):
		
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
	def _get_roi(self, image, y1, y2, x1, x2):
		
		return image[y1:y2, x1:x2]

	def _ssim_compare(self, image1, image2):

		score, diff = compare_ssim(image1, image2, full=True)
		return score, diff

	def _hist_compare(self, image1, image2):
		
		h1 = cv2.calcHist(image1, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
		h1 = cv2.normalize(h1,h1).flatten()
		h2 = cv2.calcHist(image2, [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
		h2 = cv2.normalize(h2,h2).flatten()
		return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
		
		
	def begin_compare(self, data_folder, starttime, snapcount, roi=None):
		#Returning a bunch of stuff now for monitoring purposes
		comparison = dict(
					type="compare-notify",
        			frommean=0.0,
        			ssimscore=0.0,
        			mean=0.0,
        			sd=0.0,
        			image=None,
        			badframe=False,
        			over=False,
        			added=True,
        			start=starttime,
        			data=data_folder
        			)
			
		if snapcount < self.start_count:
			return comparison
		#This is getting called after snapcount is incremented
		current_image, imagepath = self._read_image(data_folder, starttime, snapcount-1)
		comparison["image"] = imagepath
		previous_image, firstpath = self._read_image(data_folder, starttime, snapcount-2)
		
		#Preping for doing roi's
		if roi:
			current_roi = self._get_roi(current_image, 120, 692, 339, 976)
			previous_roi = self._get_roi(previous_image, 120, 692, 339, 976)
		else:
			current_roi = current_image
			previous_roi = previous_image
			
		current_roi = self._to_gray(current_roi)
		previous_roi = self._to_gray(previous_roi)
		score, diff = self._ssim_compare(current_roi, previous_roi)
		
		comparison["ssimscore"] = score
		
		if len(self.scores) > self.begin_analysis_count:
			mean = np.mean(self.scores)
			stddev = np.std(self.scores)
			diff_mean = abs(score - mean)
			from_mean = diff_mean/stddev
			comparison["frommean"] =  "{0:.3f}".format(from_mean)
			comparison["sd"] = "{0:.4f}".format(stddev)
			comparison["mean"] = "{0:.4f}".format(mean)

        		if from_mean >= self.sd_badframe_thresh:
        			comparison["badframe"] = True
        			comparison["added"] = False
        		
        		if from_mean < self.sd_badframe_thresh and from_mean > self.sd_over_thresh:       			
        			comparison["over"] = True
        			comparison["added"] = False
        			#Write difference image so we can inspect what is happening
        			diff = (diff * 255).astype("uint8")
        			diffpath = utility.get_snapshot_temp_directory(data_folder)
        			diffpath = "{0}difference_{1}.jpg".format(diffpath, snapcount) 
        			cv2.imwrite(diffpath, diff)
        		
        	if comparison["added"]:
        		self.scores.append(score)
        	
		return comparison

class FrameROI(object):
	
	def __init__(self, y, x, obj):
		self.scores = []
		