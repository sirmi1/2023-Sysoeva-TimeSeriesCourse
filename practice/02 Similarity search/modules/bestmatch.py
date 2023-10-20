import numpy as np
import copy

from modules.utils import *
from modules.metrics import *


class BestMatchFinder:
    """
    Base Best Match Finder.
    
    Parameters
    ----------
    query : numpy.ndarrray
        Query.
    
    ts_data : numpy.ndarrray
        Time series.
    
    excl_zone_denom : float, default = 1
        The exclusion zone.
    
    top_k : int, default = 3
        Count of the best match subsequences.
    
    normalize : bool, default = True
        Z-normalize or not subsequences before computing distances.
    
    r : float, default = 0.05
        Warping window size.
    """

    def __init__(self, ts_data, query, exclusion_zone=1, top_k=3, normalize=True, r=0.05):

        self.query = copy.deepcopy(np.array(query))
        if (len(ts_data.shape) == 2): # time series set
            self.ts_data = ts_data
        else:
            self.ts_data = sliding_window(ts_data, len(query))

        self.excl_zone_denom = exclusion_zone
        self.top_k = top_k
        self.normalize = normalize
        self.r = r


    def _apply_exclusion_zone(self, a, idx, excl_zone):
        """
        Apply an exclusion zone to an array (inplace).
        
        Parameters
        ----------
        a : numpy.ndarrray
            The array to apply the exclusion zone to.
        
        idx : int
            The index around which the window should be centered.
        
        excl_zone : int
            Size of the exclusion zone.
        
        Returns
        -------
        a: numpy.ndarrray
            The array which is applied the exclusion zone.
        """
        
        zone_start = max(0, idx - excl_zone)
        zone_stop = min(a.shape[-1], idx + excl_zone)
        a[zone_start : zone_stop + 1] = np.inf

        return a


    def _top_k_match(self, distances, m, bsf, excl_zone):
        """
        Find the top-k match subsequences.
        
        Parameters
        ----------
        distances : list
            Distances between query and subsequences of time series.
        
        m : int
            Subsequence length.
        
        bsf : float
            Best-so-far.
        
        excl_zone : int
            Size of the exclusion zone.
        
        Returns
        -------
        best_match_results: dict
            Dictionary containing results of algorithm.
        """
        
        data_len = len(distances)
        top_k_match = []

        distances = np.copy(distances)
        top_k_match_idx = []
        top_k_match_dist = []

        for i in range(self.top_k):
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > bsf):
                break

            distances = self._apply_exclusion_zone(distances, min_idx, excl_zone)

            top_k_match_idx.append(min_idx)
            top_k_match_dist.append(min_dist)

        return {'index': top_k_match_idx, 'distance': top_k_match_dist}


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder.
    """
    
    def __init__(self, ts=None, query=None, exclusion_zone=1, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)


    def perform(self):
        """
        Выполните поиск лучшего совпадения, используя наивный алгоритм.
        
        Returns
        -------
        best_match_results: dict
             Словарь, содержащий результаты наивного алгоритма.
        """ 
        N, m = self.ts_data.shape
        
        bsf = float("inf")
        
        if (self.excl_zone_denom is None):
            excl_zone = 0
        else:
            excl_zone = int(np.ceil(m / self.excl_zone_denom))
        
        dist_list = []
        query_data = z_normalize(self.query)
        for i in range(N):
          query_subsequence = self.ts_data[i]
          query_subsequence = z_normalize(query_subsequence)
          dist= DTW_distance(query_subsequence, query_data)

          if dist < bsf:
            dist_list.append(dist)
            self.bestmatch = super()._top_k_match(dist_list, m, bsf, excl_zone )
            if len(self.bestmatch['index'])==self.top_k:
              bsf = self.bestmatch['distance'][self.top_k-1]
          else:
            dist_list.append(np.inf)

        return self.bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder.
    """
    
    def __init__(self, ts=None, query=None, exclusion_zone=1, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)


    def _LB_Kim(self, subs1, subs2):
        """
        Compute LB_Kim lower bound between two subsequences.
        
        Parameters
        ----------
        subs1 : numpy.ndarrray
            The first subsequence.
        
        subs2 : numpy.ndarrray
            The second subsequence.
        
        Returns
        -------
        lb_Kim : float
            LB_Kim lower bound.
        """

        if len(subs1) != len(subs2):
          raise ValueError("Подпоследовательности должны иметь одинаковую длину.")
    
        lb_Kim = 0
    
        lb_Kim = (subs1[0] - subs2[0])**2 + (subs1[-1] - subs2[-1])**2

        return lb_Kim

    def _LB_Keogh(self, subs1, subs2, r):
        """
        Compute LB_Keogh lower bound between two subsequences.
        
        Parameters
        ----------
        subs1 : numpy.ndarrray
            The first subsequence.
        
        subs2 : numpy.ndarrray
            The second subsequence.
        
        r : float
            Warping window size.
        
        Returns
        -------
        lb_Keogh : float
            LB_Keogh lower bound.
        """
        
        lb_Keogh = 0

        if len(subs1) != len(subs2):
          raise ValueError("Подпоследовательности должны иметь одинаковую длину.")
    
        res = []
        for i in range(len(subs1)):
            if subs2[i] > max(subs1[max(0, i-r):min(len(subs1), i+r)]):
              res.append((subs2[i] - max(subs1[max(0, i-r):min(len(subs1), i+r)]))**2)
            elif subs2[i] < min(subs1[max(0, i-r):min(len(subs1), i+r)]):
              res.append((subs2[i] - min(subs1[max(0, i-r):min(len(subs1), i+r)]))**2)
            else:
              res.append(0)

        lb_Keogh = sum(res)

        return lb_Keogh


    def perform(self):
        """
        Perform the best match finder using UCR-DTW algorithm.
        
        Returns
        -------
        best_match_results: dict
            Dictionary containing results of UCR-DTW algorithm.
        """
        N, m = self.ts_data.shape
        
        bsf = float("inf")
        
        if (self.excl_zone_denom is None):
            excl_zone = 0
        else:
            excl_zone = int(np.ceil(m / self.excl_zone_denom))
        
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0
        
        distances = []

        if self.normalize:
            self.query = z_normalize(self.query)
 
        for subseq_idx in range(N):
          subseq = self.ts_data[subseq_idx]
     
          if self.normalize:
            subseq = z_normalize(subseq)

          if self._LB_Kim(self.query, subseq) > bsf:
            self.lb_Kim_num += 1
            print(self._LB_Kim(self.query, subseq))
            distances.append(np.inf)

          elif self._LB_Keogh(self.query, subseq, int(self.r*m)) > bsf:
            self.lb_KeoghQC_num += 1
            distances.append(np.inf)

          elif self._LB_Keogh(subseq, self.query, int(self.r*m)) > bsf:
            self.lb_KeoghCQ_num += 1
            distances.append(np.inf)

          else:
            dist = DTW_distance(self.query, subseq, self.r)
            #print(f'dist = {dist}')
            #print(f'bsf = {bsf}')

            if dist < bsf:
              distances.append(dist)
              self.bestmatch = super()._top_k_match(distances, m, bsf, excl_zone )
              if len(self.bestmatch['index'])==self.top_k:
                bsf = self.bestmatch['distance'][self.top_k-1]
            else:
              distances.append(np.inf)

        return {'index' : self.bestmatch['index'],
                'distance' : self.bestmatch['distance'],
                'lb_Kim_num': self.lb_Kim_num,
                'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
                'lb_KeoghQC_num': self.lb_KeoghQC_num
                }
