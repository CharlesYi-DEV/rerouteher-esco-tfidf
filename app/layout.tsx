import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ReRouteHer CV → ESCO Feasibility Test',
  description: 'Internal CV upload test comparing TF-IDF and MiniLM ESCO matching.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
