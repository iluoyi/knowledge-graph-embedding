package utils;

public class AvgRankHitAtTenBean {
	public long totalRawHeadRank = 0L;
	public long totalRawTailRank = 0L;
	public long totalFilterHeadRank = 0L;
	public long totalFilterTailRank = 0L;
	
	public long totalRawHeadHit10 = 0L;
	public long totalRawTailHit10 = 0L;
	public long totalFilterHeadHit10 = 0L;
	public long totalFilterTailHit10 = 0L;
	
	public static AvgRankHitAtTenBean merge(AvgRankHitAtTenBean r1, AvgRankHitAtTenBean r2) {
		AvgRankHitAtTenBean r3 = new AvgRankHitAtTenBean();
		r3.totalRawHeadRank = r1.totalRawHeadRank + r2.totalRawHeadRank;
		r3.totalRawTailRank = r1.totalRawTailRank + r2.totalRawTailRank;
		r3.totalFilterHeadRank = r1.totalFilterHeadRank + r2.totalFilterHeadRank;
		r3.totalFilterTailRank = r1.totalFilterTailRank + r2.totalFilterTailRank;
		
		r3.totalRawHeadHit10 = r1.totalRawHeadHit10 + r2.totalRawHeadHit10;
		r3.totalRawTailHit10 = r1.totalRawTailHit10 + r2.totalRawTailHit10;
		r3.totalFilterHeadHit10 = r1.totalFilterHeadHit10 + r2.totalFilterHeadHit10;
		r3.totalFilterTailHit10 = r1.totalFilterTailHit10 + r2.totalFilterTailHit10;
		return r3;
	}
}
