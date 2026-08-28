import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ReRouteHer CV → 6-digit MASCO Matching Test',
  description: 'Internal CV upload test comparing JobHop-trained TF-IDF and MiniLM matching to exact six-digit MASCO occupations.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
