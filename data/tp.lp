\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: Trailer Optimization Model

Minimize
 obj:
Subject To
 c1: s_('97624',_'out')_src_pool + s_('97045',_'out')_src_pool
     + s_('98841',_'out')_src_pool + s_('89801',_'out')_src_pool - Flow_value
      = 0
 c2: ('97624',_'in')_t_sink_pool + ('97045',_'in')_t_sink_pool
     + ('98841',_'in')_t_sink_pool + ('89801',_'in')_t_sink_pool - Flow_value
      = 0
 c3: - s_('97624',_'out')_src_pool + ('97624',_'out')_(1,_0,_'start')_trl_supp
     + ('97624',_'out')_(2,_0,_'start')_trl_supp
     + ('97624',_'out')_(3,_0,_'start')_trl_supp = 0
 c4: s_('97624',_'out')_src_pool - ('97624',_'in')_t_sink_pool = 0
 c5: ('97624',_'in')_t_sink_pool - (1,_1,_'end')_('97624',_'in')_trl_rtrn
     - (2,_1,_'end')_('97624',_'in')_trl_rtrn
     - (3,_1,_'end')_('97624',_'in')_trl_rtrn = 0
 c6: - s_('97045',_'out')_src_pool + ('97045',_'out')_(1,_1,_'start')_trl_supp
      = 0
 c7: s_('97045',_'out')_src_pool - ('97045',_'in')_t_sink_pool = 0
 c8: ('97045',_'in')_t_sink_pool - (1,_0,_'end')_('97045',_'in')_trl_rtrn = 0
 c9: - s_('98841',_'out')_src_pool + ('98841',_'out')_(2,_1,_'start')_trl_supp
      = 0
 c10: s_('98841',_'out')_src_pool - ('98841',_'in')_t_sink_pool = 0
 c11: ('98841',_'in')_t_sink_pool - (2,_0,_'end')_('98841',_'in')_trl_rtrn = 0
 c12: - s_('89801',_'out')_src_pool + ('89801',_'out')_(3,_1,_'start')_trl_supp
       = 0
 c13: s_('89801',_'out')_src_pool - ('89801',_'in')_t_sink_pool = 0
 c14: ('89801',_'in')_t_sink_pool - (3,_0,_'end')_('89801',_'in')_trl_rtrn = 0
 c15: - ('97624',_'out')_(1,_0,_'start')_trl_supp
      + (1,_0,_'start')_(1,_0,_'end')_trailer_utilization = 0
 c16: - ('97045',_'out')_(1,_1,_'start')_trl_supp
      + (1,_1,_'start')_(1,_1,_'end')_trailer_utilization = 0
 c17: - ('97624',_'out')_(2,_0,_'start')_trl_supp
      + (2,_0,_'start')_(2,_0,_'end')_trailer_utilization
      - (1,_1,_'end')_(2,_0,_'start')_same_trailer
      - (3,_1,_'end')_(2,_0,_'start')_same_trailer = 0
 c18: - ('98841',_'out')_(2,_1,_'start')_trl_supp
      + (2,_1,_'start')_(2,_1,_'end')_trailer_utilization = 0
 c19: - ('97624',_'out')_(3,_0,_'start')_trl_supp
      + (3,_0,_'start')_(3,_0,_'end')_trailer_utilization = 0
 c20: - ('89801',_'out')_(3,_1,_'start')_trl_supp
      + (3,_1,_'start')_(3,_1,_'end')_trailer_utilization = 0
 c21: - (1,_0,_'start')_(1,_0,_'end')_trailer_utilization
      + (1,_0,_'end')_('97045',_'in')_trl_rtrn = 0
 c22: - (1,_1,_'start')_(1,_1,_'end')_trailer_utilization
      + (1,_1,_'end')_('97624',_'in')_trl_rtrn
      + (1,_1,_'end')_(2,_0,_'start')_same_trailer = 0
 c23: - (2,_0,_'start')_(2,_0,_'end')_trailer_utilization
      + (2,_0,_'end')_('98841',_'in')_trl_rtrn = 0
 c24: - (2,_1,_'start')_(2,_1,_'end')_trailer_utilization
      + (2,_1,_'end')_('97624',_'in')_trl_rtrn = 0
 c25: - (3,_0,_'start')_(3,_0,_'end')_trailer_utilization
      + (3,_0,_'end')_('89801',_'in')_trl_rtrn = 0
 c26: - (3,_1,_'start')_(3,_1,_'end')_trailer_utilization
      + (3,_1,_'end')_('97624',_'in')_trl_rtrn
      + (3,_1,_'end')_(2,_0,_'start')_same_trailer = 0

Bounds
       (1,_0,_'start')_(1,_0,_'end')_trailer_utilization = 1
       (1,_1,_'start')_(1,_1,_'end')_trailer_utilization = 1
       (2,_0,_'start')_(2,_0,_'end')_trailer_utilization = 1
       (2,_1,_'start')_(2,_1,_'end')_trailer_utilization = 1
       (3,_0,_'start')_(3,_0,_'end')_trailer_utilization = 1
       (3,_1,_'start')_(3,_1,_'end')_trailer_utilization = 1
End
