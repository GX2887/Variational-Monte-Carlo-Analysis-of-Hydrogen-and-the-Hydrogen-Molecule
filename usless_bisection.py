def find_bisec(upb, lowb):
    bi_sec = 0.5*(upb+lowb)
    pd_bi = pd_H_theta(bi_sec)
    return bi_sec, pd_bi

def bisec_root_finding(upb, lowb, tol=1e-8):
    diff = 1 # set a stopping condition for the loop
    while diff > tol:
        bi_sec, pd_bi = find_bisec(upb, lowb)
        diff = np.abs(pd_bi-0) # update the stopping condition
        if pd_H_theta(upb)*pd_bi<0:
            lowb = bi_sec
        if pd_H_theta(lowb)*pd_bi<0:
            upb = bi_sec
        print(bi_sec)
    return bi_sec # theta of lowest energy

# mini_theta = bisec_root_finding(0.8, 1.1)